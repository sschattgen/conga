"""
Comprehensive performance benchmarking infrastructure for FAISS vs sklearn neighbor search.

This module provides tools to measure and compare the performance of different backends
(FAISS-GPU, FAISS-CPU, sklearn) for both GEX and TCR neighbor search operations.
It includes memory usage profiling, performance scaling analysis, and backend
selection recommendations.

Key components:
- BenchmarkResult: Dataclass for storing benchmark measurements
- PerformanceSuite: Main benchmarking interface
- Backend comparison and scaling analysis
- Memory usage profiling
- Accuracy validation between backends

Example usage:
    suite = PerformanceSuite()
    results = suite.run_gex_benchmark(X_gex, nbr_fracs=[0.01, 0.05])
    suite.generate_scaling_curves(results)
    recommendations = suite.get_backend_recommendations(results)
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from pathlib import Path

# Memory profiling
import psutil
import gc

# Scientific computing
from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_blobs

# CoNGA imports
from . import util
from .neighbors import (
    FaissNeighborSearcher, 
    Backend, 
    get_backend_info,
    compute_neighbor_distances,
    compute_tcr_vector_neighbors
)

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Container for benchmark measurements."""
    backend: str
    data_type: str  # 'gex' or 'tcr'
    n_samples: int
    n_features: int
    nbr_fracs: List[float]
    query_time: float  # seconds
    index_time: float  # seconds (for FAISS index building)
    peak_memory_mb: float
    memory_efficiency: float  # MB per 1000 samples
    neighbors: Dict[float, np.ndarray]  # actual neighbor results
    accuracy_vs_baseline: Optional[float] = None  # fraction of identical neighbors
    notes: str = ""

@dataclass  
class ScalingResult:
    """Results from scaling analysis across dataset sizes."""
    backend: str
    data_type: str
    sample_sizes: List[int]
    query_times: List[float]
    memory_usage: List[float] 
    speedup_vs_sklearn: List[float]
    notes: str = ""
class MemoryTracker:
    """Context manager for tracking memory usage during operations."""
    
    def __init__(self, description: str = "operation"):
        self.description = description
        self.start_memory = 0
        self.peak_memory = 0
        self.process = psutil.Process()
        
    def __enter__(self):
        gc.collect()  # Clean up before measurement
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = max(self.start_memory, current_memory)
        logger.debug(f"{self.description}: {self.peak_memory - self.start_memory:.1f} MB peak")
        
    def get_peak_memory_mb(self) -> float:
        """Get peak memory usage in MB."""
        return self.peak_memory - self.start_memory


class DatasetGenerator:
    """Generate synthetic datasets for benchmarking."""
    
    @staticmethod
    def generate_gex_data(n_samples: int, n_features: int = 2000, 
                          random_seed: int = 42) -> np.ndarray:
        """Generate synthetic GEX data matrix."""
        np.random.seed(random_seed)
        
        # Use make_blobs to create realistic GEX-like clustered data
        X, _ = make_blobs(
            n_samples=n_samples,
            n_features=n_features, 
            centers=max(1, n_samples // 500),  # Realistic cluster count
            cluster_std=2.0,
            center_box=(-10.0, 10.0),
            random_state=random_seed
        )
        
        # Add some noise and make it resemble log-normalized gene expression
        X = np.abs(X) + np.random.exponential(0.1, X.shape)
        X = np.log1p(X)  # Log transform like real scRNA-seq data
        
        return X.astype(np.float32)
    
    @staticmethod
    def generate_tcr_vector_data(n_samples: int, vector_length: int = 1136,
                                 random_seed: int = 42) -> np.ndarray:
        """Generate synthetic vectorized TCR data matrix."""
        np.random.seed(random_seed)
        
        # TCR vectors are more structured than random - simulate blocks
        n_germline = vector_length - 120  # Germline portion
        n_cdr3 = 120  # CDR3 portion
        
        # Germline portion: lower variance, more discrete
        X_germline = np.random.gamma(2, 0.5, (n_samples, n_germline))
        
        # CDR3 portion: higher variance, more continuous  
        X_cdr3 = np.random.normal(0, 1, (n_samples, n_cdr3))
        
        X = np.hstack([X_germline, X_cdr3]).astype(np.float32)
        
        # Add some clustering structure
        n_clusters = max(1, n_samples // 100)
        cluster_centers = np.random.normal(0, 2, (n_clusters, vector_length))
        cluster_assignments = np.random.randint(0, n_clusters, n_samples)
        
        for i, cluster in enumerate(cluster_assignments):
            X[i] += 0.3 * cluster_centers[cluster]
            
        return X
    
    @staticmethod
    def generate_exclude_groups(n_samples: int, 
                                random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic TCR alpha/beta groups for exclusion testing."""
        np.random.seed(random_seed)
        
        # Create realistic clonotype grouping
        n_clonotypes = max(1, n_samples // 3)  # ~3 cells per clonotype on average
        
        agroups = np.random.randint(0, n_clonotypes, n_samples)
        bgroups = np.random.randint(0, n_clonotypes, n_samples)
        
        return agroups, bgroups
class PerformanceSuite:
    """
    Comprehensive benchmarking suite for FAISS vs sklearn neighbor search.
    
    Provides tools to measure query time, memory usage, and accuracy across
    different backends and dataset sizes for both GEX and TCR data types.
    """
    
    def __init__(self, max_memory_gb: float = 8.0):
        """
        Initialize benchmarking suite.
        
        Parameters:
        -----------
        max_memory_gb : float
            Maximum memory limit for large dataset tests (default: 8GB)
        """
        self.max_memory_gb = max_memory_gb
        self.backend_info = get_backend_info()
        self.results: List[BenchmarkResult] = []
        
        logger.info(f"Benchmark suite initialized. Available backends: {self.backend_info}")
    
    def benchmark_single_dataset(
        self,
        X: np.ndarray,
        data_type: str,
        nbr_fracs: List[float],
        exclude_groups: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        test_backends: Optional[List[str]] = None,
        validate_accuracy: bool = True
    ) -> List[BenchmarkResult]:
        """
        Benchmark neighbor search on a single dataset across multiple backends.
        
        Parameters:
        -----------
        X : np.ndarray
            Data matrix to benchmark (samples x features)
        data_type : str
            Type of data ('gex' or 'tcr')  
        nbr_fracs : List[float]
            Neighbor fractions to test
        exclude_groups : Optional[Tuple[np.ndarray, np.ndarray]]
            Alpha/beta groups for TCR exclusion testing
        test_backends : Optional[List[str]]
            Backends to test (default: all available)
        validate_accuracy : bool
            Whether to validate accuracy against sklearn baseline
            
        Returns:
        --------
        List[BenchmarkResult]
            Results for each backend tested
        """
        if test_backends is None:
            test_backends = ['sklearn']
            if self.backend_info['faiss_cpu_available']:
                test_backends.append('faiss-cpu')
            if self.backend_info['faiss_gpu_available']:
                test_backends.append('faiss-gpu')
        
        results = []
        baseline_neighbors = None
        
        # Test each backend
        for backend_name in test_backends:
            logger.info(f"Benchmarking {backend_name} on {data_type} data: {X.shape}")
            
            try:
                result = self._benchmark_backend(
                    X=X,
                    data_type=data_type,
                    backend_name=backend_name,
                    nbr_fracs=nbr_fracs,
                    exclude_groups=exclude_groups
                )
                
                # Store baseline for accuracy comparison
                if backend_name == 'sklearn' and validate_accuracy:
                    baseline_neighbors = result.neighbors
                elif baseline_neighbors is not None and validate_accuracy:
                    # Compare accuracy against sklearn baseline
                    result.accuracy_vs_baseline = self._calculate_accuracy(
                        result.neighbors, baseline_neighbors, nbr_fracs
                    )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Backend {backend_name} failed: {e}")
                # Create error result
                results.append(BenchmarkResult(
                    backend=backend_name,
                    data_type=data_type,
                    n_samples=X.shape[0],
                    n_features=X.shape[1],
                    nbr_fracs=nbr_fracs,
                    query_time=float('inf'),
                    index_time=float('inf'),
                    peak_memory_mb=float('inf'),
                    memory_efficiency=float('inf'),
                    neighbors={},
                    notes=f"Failed: {str(e)}"
                ))
        
        self.results.extend(results)
        return results
    def _benchmark_backend(
        self,
        X: np.ndarray,
        data_type: str,
        backend_name: str,
        nbr_fracs: List[float],
        exclude_groups: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> BenchmarkResult:
        """Benchmark a specific backend on given data."""
        
        # Convert backend name to Backend enum for FAISS
        backend_enum = None
        if backend_name == 'faiss-gpu':
            backend_enum = Backend.FAISS_GPU
        elif backend_name == 'faiss-cpu':
            backend_enum = Backend.FAISS_CPU
        elif backend_name == 'sklearn':
            backend_enum = Backend.SKLEARN
        
        # Measure index building time (for FAISS)
        index_start = time.time()
        
        # Create searcher with forced backend
        if backend_enum in [Backend.FAISS_GPU, Backend.FAISS_CPU]:
            searcher = FaissNeighborSearcher(force_backend=backend_enum)
        else:
            searcher = None  # Will use direct sklearn implementation
        
        index_time = time.time() - index_start
        
        # Measure query time and memory
        with MemoryTracker(f"{backend_name} {data_type} search") as memory:
            query_start = time.time()
            
            if data_type == 'gex':
                if searcher is not None:
                    # Use FAISS
                    result = searcher.search_neighbors(
                        X=X,
                        nbr_fracs=nbr_fracs,
                        exclude_groups=exclude_groups,
                        metric='euclidean',
                        data_type='gex'
                    )
                    neighbors = result.neighbors
                else:
                    # Use direct sklearn (baseline)
                    neighbors = self._sklearn_gex_search(X, nbr_fracs, exclude_groups)
                    
            elif data_type == 'tcr':
                if searcher is not None:
                    # Use FAISS for TCR vectors
                    result = searcher.search_neighbors(
                        X=X,
                        nbr_fracs=nbr_fracs,
                        exclude_groups=exclude_groups,
                        metric='euclidean',  # Use euclidean for TCR vectors (L2 squared internally)
                        data_type='tcr'
                    )
                    neighbors = result.neighbors
                else:
                    # Use direct sklearn (baseline) 
                    neighbors = self._sklearn_tcr_search(X, nbr_fracs, exclude_groups)
            else:
                raise ValueError(f"Unknown data type: {data_type}")
            
            query_time = time.time() - query_start
        
        # Calculate memory efficiency
        peak_memory = memory.get_peak_memory_mb()
        memory_efficiency = peak_memory / (X.shape[0] / 1000)  # MB per 1000 samples
        
        return BenchmarkResult(
            backend=backend_name,
            data_type=data_type,
            n_samples=X.shape[0],
            n_features=X.shape[1],
            nbr_fracs=nbr_fracs,
            query_time=query_time,
            index_time=index_time,
            peak_memory_mb=peak_memory,
            memory_efficiency=memory_efficiency,
            neighbors=neighbors,
            notes=f"Success"
        )
    
    def _sklearn_gex_search(
        self, 
        X: np.ndarray, 
        nbr_fracs: List[float],
        exclude_groups: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[float, np.ndarray]:
        """Direct sklearn neighbor search for GEX data (baseline)."""
        
        D = pairwise_distances(X, metric='euclidean')
        
        # Apply exclusions if provided
        if exclude_groups is not None:
            agroups, bgroups = exclude_groups
            for ii, (a, b) in enumerate(zip(agroups, bgroups)):
                D[ii, (agroups == a)] = 1e3
                D[ii, (bgroups == b)] = 1e3
        
        neighbors = {}
        for nbr_frac in nbr_fracs:
            num_neighbors = max(1, int(nbr_frac * X.shape[0]))
            nbrs = np.argpartition(D, num_neighbors - 1)[:, :num_neighbors]
            neighbors[nbr_frac] = nbrs
            
        return neighbors
    
    def _sklearn_tcr_search(
        self, 
        X: np.ndarray, 
        nbr_fracs: List[float],
        exclude_groups: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[float, np.ndarray]:
        """Direct sklearn neighbor search for TCR vectors (baseline)."""
        
        # Use squared euclidean for TCR vectors
        D = pairwise_distances(X, metric='sqeuclidean')
        
        # Apply exclusions if provided  
        if exclude_groups is not None:
            agroups, bgroups = exclude_groups
            for ii, (a, b) in enumerate(zip(agroups, bgroups)):
                D[ii, (agroups == a)] = 1e6  # Large value
                D[ii, (bgroups == b)] = 1e6
        
        neighbors = {}
        for nbr_frac in nbr_fracs:
            num_neighbors = max(1, int(nbr_frac * X.shape[0]))
            nbrs = np.argpartition(D, num_neighbors - 1)[:, :num_neighbors]
            neighbors[nbr_frac] = nbrs
            
        return neighbors
    def _calculate_accuracy(
        self,
        test_neighbors: Dict[float, np.ndarray],
        baseline_neighbors: Dict[float, np.ndarray], 
        nbr_fracs: List[float]
    ) -> float:
        """Calculate accuracy as fraction of identical neighbors vs baseline."""
        
        total_matches = 0
        total_neighbors = 0
        
        for nbr_frac in nbr_fracs:
            test_nbrs = test_neighbors[nbr_frac]
            baseline_nbrs = baseline_neighbors[nbr_frac]
            
            # Count matches (order doesn't matter for neighbor sets)
            for i in range(test_nbrs.shape[0]):
                test_set = set(test_nbrs[i])
                baseline_set = set(baseline_nbrs[i])
                matches = len(test_set.intersection(baseline_set))
                total_matches += matches
                total_neighbors += len(test_set)
        
        return total_matches / total_neighbors if total_neighbors > 0 else 0.0
    
    def run_scaling_benchmark(
        self,
        data_type: str,
        sample_sizes: List[int],
        n_features: Optional[int] = None,
        nbr_fracs: List[float] = [0.01, 0.05],
        test_backends: Optional[List[str]] = None,
        random_seed: int = 42
    ) -> List[ScalingResult]:
        """
        Run scaling benchmark across different dataset sizes.
        
        Parameters:
        -----------
        data_type : str
            Type of data to generate ('gex' or 'tcr')
        sample_sizes : List[int]
            Sample sizes to test
        n_features : Optional[int]
            Number of features (default: 2000 for GEX, 1136 for TCR)
        nbr_fracs : List[float] 
            Neighbor fractions to test
        test_backends : Optional[List[str]]
            Backends to test (default: all available)
        random_seed : int
            Random seed for data generation
            
        Returns:
        --------
        List[ScalingResult]
            Scaling results for each backend
        """
        
        if n_features is None:
            n_features = 2000 if data_type == 'gex' else 1136
            
        if test_backends is None:
            test_backends = ['sklearn']
            if self.backend_info['faiss_cpu_available']:
                test_backends.append('faiss-cpu')
            if self.backend_info['faiss_gpu_available']:
                test_backends.append('faiss-gpu')
        
        # Initialize results structure
        scaling_results = {backend: ScalingResult(
            backend=backend,
            data_type=data_type,
            sample_sizes=[],
            query_times=[],
            memory_usage=[],
            speedup_vs_sklearn=[]
        ) for backend in test_backends}
        
        sklearn_times = {}  # For speedup calculation
        
        for n_samples in sample_sizes:
            logger.info(f"Scaling test: {data_type} data with {n_samples} samples")
            
            # Skip if memory estimate exceeds limit
            estimated_memory_gb = (n_samples * n_features * 8) / (1024**3)  # float64
            if estimated_memory_gb > self.max_memory_gb:
                logger.warning(f"Skipping {n_samples} samples (estimated {estimated_memory_gb:.1f} GB > {self.max_memory_gb} GB)")
                continue
            
            # Generate test data
            if data_type == 'gex':
                X = DatasetGenerator.generate_gex_data(n_samples, n_features, random_seed)
            elif data_type == 'tcr':
                X = DatasetGenerator.generate_tcr_vector_data(n_samples, n_features, random_seed)
            else:
                raise ValueError(f"Unknown data type: {data_type}")
            
            # Generate exclusion groups for testing
            exclude_groups = DatasetGenerator.generate_exclude_groups(n_samples, random_seed)
            
            # Benchmark each backend
            results = self.benchmark_single_dataset(
                X=X,
                data_type=data_type,
                nbr_fracs=nbr_fracs,
                exclude_groups=exclude_groups,
                test_backends=test_backends,
                validate_accuracy=False  # Skip accuracy for scaling tests
            )
            
            # Extract scaling metrics
            for result in results:
                backend = result.backend
                if backend in scaling_results:
                    scaling_results[backend].sample_sizes.append(n_samples)
                    scaling_results[backend].query_times.append(result.query_time)
                    scaling_results[backend].memory_usage.append(result.peak_memory_mb)
                    
                    # Store sklearn times for speedup calculation
                    if backend == 'sklearn':
                        sklearn_times[n_samples] = result.query_time
            
            # Calculate speedups vs sklearn
            if n_samples in sklearn_times:
                sklearn_time = sklearn_times[n_samples]
                for backend in test_backends:
                    if backend != 'sklearn' and backend in scaling_results:
                        backend_results = scaling_results[backend]
                        if len(backend_results.query_times) > 0:
                            backend_time = backend_results.query_times[-1]
                            speedup = sklearn_time / backend_time if backend_time > 0 else float('inf')
                            backend_results.speedup_vs_sklearn.append(speedup)
        
        return list(scaling_results.values())
    def run_gex_benchmark(
        self,
        X_gex: np.ndarray,
        nbr_fracs: List[float] = [0.01, 0.05, 0.10],
        exclude_groups: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> List[BenchmarkResult]:
        """
        Run comprehensive GEX neighbor search benchmark.
        
        Parameters:
        -----------
        X_gex : np.ndarray
            GEX data matrix (cells x genes)
        nbr_fracs : List[float]
            Neighbor fractions to test
        exclude_groups : Optional[Tuple[np.ndarray, np.ndarray]]
            Alpha/beta groups for exclusion testing
            
        Returns:
        --------
        List[BenchmarkResult]
            Results for each backend tested
        """
        return self.benchmark_single_dataset(
            X=X_gex,
            data_type='gex',
            nbr_fracs=nbr_fracs,
            exclude_groups=exclude_groups,
            validate_accuracy=True
        )
    
    def run_tcr_benchmark(
        self,
        X_tcr: np.ndarray,
        nbr_fracs: List[float] = [0.01, 0.05, 0.10],
        exclude_groups: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> List[BenchmarkResult]:
        """
        Run comprehensive TCR vector neighbor search benchmark.
        
        Parameters:
        -----------
        X_tcr : np.ndarray
            Vectorized TCR data matrix (clonotypes x vector_length)
        nbr_fracs : List[float]
            Neighbor fractions to test
        exclude_groups : Optional[Tuple[np.ndarray, np.ndarray]]
            Alpha/beta groups for exclusion testing
            
        Returns:
        --------
        List[BenchmarkResult]
            Results for each backend tested
        """
        return self.benchmark_single_dataset(
            X=X_tcr,
            data_type='tcr',
            nbr_fracs=nbr_fracs,
            exclude_groups=exclude_groups,
            validate_accuracy=True
        )
    
    def generate_performance_report(
        self, 
        results: Optional[List[BenchmarkResult]] = None
    ) -> pd.DataFrame:
        """
        Generate comprehensive performance report.
        
        Parameters:
        -----------
        results : Optional[List[BenchmarkResult]]
            Results to include (default: all collected results)
            
        Returns:
        --------
        pd.DataFrame
            Performance report with metrics for each backend/dataset combination
        """
        if results is None:
            results = self.results
            
        if not results:
            return pd.DataFrame()
        
        report_data = []
        for result in results:
            # Calculate speedup vs baseline sklearn if available
            sklearn_baseline = None
            for r in results:
                if (r.backend == 'sklearn' and 
                    r.data_type == result.data_type and 
                    r.n_samples == result.n_samples):
                    sklearn_baseline = r
                    break
            
            speedup = (sklearn_baseline.query_time / result.query_time 
                      if sklearn_baseline and result.query_time > 0 
                      else None)
            
            report_data.append({
                'backend': result.backend,
                'data_type': result.data_type,
                'n_samples': result.n_samples,
                'n_features': result.n_features,
                'query_time_sec': result.query_time,
                'index_time_sec': result.index_time,
                'peak_memory_mb': result.peak_memory_mb,
                'memory_efficiency_mb_per_1k': result.memory_efficiency,
                'speedup_vs_sklearn': speedup,
                'accuracy_vs_baseline': result.accuracy_vs_baseline,
                'notes': result.notes
            })
        
        df = pd.DataFrame(report_data)
        
        # Sort by data type, sample size, then by speedup
        if not df.empty:
            df = df.sort_values(['data_type', 'n_samples', 'speedup_vs_sklearn'], 
                              ascending=[True, True, False])
        
        return df
    def get_backend_recommendations(
        self, 
        results: Optional[List[BenchmarkResult]] = None
    ) -> Dict[str, str]:
        """
        Generate backend selection recommendations based on benchmark results.
        
        Parameters:
        -----------
        results : Optional[List[BenchmarkResult]]
            Results to analyze (default: all collected results)
            
        Returns:
        --------
        Dict[str, str]
            Recommendations for each data type and use case
        """
        if results is None:
            results = self.results
            
        recommendations = {}
        
        # Group results by data type and sample size
        grouped = {}
        for result in results:
            key = (result.data_type, result.n_samples)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)
        
        # Analyze each group
        for (data_type, n_samples), group_results in grouped.items():
            # Find best performer (fastest with acceptable accuracy)
            best_backend = None
            best_speedup = 0
            
            for result in group_results:
                if result.notes == "Success":
                    # Require >95% accuracy if available
                    if result.accuracy_vs_baseline is None or result.accuracy_vs_baseline >= 0.95:
                        # Calculate speedup vs sklearn
                        sklearn_time = None
                        for r in group_results:
                            if r.backend == 'sklearn':
                                sklearn_time = r.query_time
                                break
                        
                        if sklearn_time and result.query_time > 0:
                            speedup = sklearn_time / result.query_time
                            if speedup > best_speedup:
                                best_speedup = speedup
                                best_backend = result.backend
            
            # Generate recommendation
            size_category = "small" if n_samples < 5000 else "medium" if n_samples < 50000 else "large"
            
            if best_backend is None:
                rec = f"No suitable backend found for {data_type} {size_category} datasets ({n_samples} samples)"
            elif best_backend == 'sklearn':
                rec = f"sklearn (baseline) - {data_type} {size_category} datasets ({n_samples} samples)"
            else:
                rec = f"{best_backend} ({best_speedup:.1f}x speedup) - {data_type} {size_category} datasets ({n_samples} samples)"
            
            recommendations[f"{data_type}_{size_category}_{n_samples}"] = rec
        
        # General recommendations
        if self.backend_info['faiss_gpu_available']:
            recommendations['gpu_available'] = "FAISS-GPU recommended for large datasets (>20k samples) when GPU memory permits"
        if self.backend_info['faiss_cpu_available']:
            recommendations['cpu_available'] = "FAISS-CPU recommended for medium-large datasets (5k-100k samples)"
        recommendations['baseline'] = "sklearn always available as fallback, identical results guaranteed"
        
        return recommendations
    
    def save_results(self, output_path: str):
        """Save benchmark results to file."""
        output_path = Path(output_path)
        
        # Save detailed results
        df = self.generate_performance_report()
        if not df.empty:
            df.to_csv(output_path.with_suffix('.csv'), index=False)
            logger.info(f"Benchmark results saved to {output_path.with_suffix('.csv')}")
        
        # Save recommendations
        recommendations = self.get_backend_recommendations()
        with open(output_path.with_suffix('.txt'), 'w') as f:
            f.write("FAISS vs sklearn Backend Recommendations\n")
            f.write("=" * 50 + "\n\n")
            for key, rec in recommendations.items():
                f.write(f"{key}: {rec}\n")
        logger.info(f"Recommendations saved to {output_path.with_suffix('.txt')}")
    
    def clear_results(self):
        """Clear collected benchmark results."""
        self.results.clear()


# Convenience functions for common benchmark scenarios

def quick_gex_benchmark(
    n_samples: int = 10000,
    n_features: int = 2000,
    nbr_fracs: List[float] = [0.01, 0.05]
) -> pd.DataFrame:
    """
    Quick GEX benchmark on synthetic data.
    
    Parameters:
    -----------
    n_samples : int
        Number of cells to generate
    n_features : int  
        Number of genes to generate
    nbr_fracs : List[float]
        Neighbor fractions to test
        
    Returns:
    --------
    pd.DataFrame
        Performance report
    """
    suite = PerformanceSuite()
    
    # Generate synthetic GEX data
    X_gex = DatasetGenerator.generate_gex_data(n_samples, n_features)
    exclude_groups = DatasetGenerator.generate_exclude_groups(n_samples)
    
    # Run benchmark
    results = suite.run_gex_benchmark(X_gex, nbr_fracs, exclude_groups)
    
    return suite.generate_performance_report(results)
def quick_tcr_benchmark(
    n_samples: int = 5000,
    vector_length: int = 1136,
    nbr_fracs: List[float] = [0.01, 0.05]
) -> pd.DataFrame:
    """
    Quick TCR benchmark on synthetic vectorized data.
    
    Parameters:
    -----------
    n_samples : int
        Number of clonotypes to generate
    vector_length : int
        TCR vector length (default: 1136 for human)
    nbr_fracs : List[float]
        Neighbor fractions to test
        
    Returns:
    --------
    pd.DataFrame
        Performance report
    """
    suite = PerformanceSuite()
    
    # Generate synthetic TCR vector data
    X_tcr = DatasetGenerator.generate_tcr_vector_data(n_samples, vector_length)
    exclude_groups = DatasetGenerator.generate_exclude_groups(n_samples)
    
    # Run benchmark
    results = suite.run_tcr_benchmark(X_tcr, nbr_fracs, exclude_groups)
    
    return suite.generate_performance_report(results)


def comprehensive_scaling_benchmark(
    output_dir: str = "benchmark_results",
    max_samples: int = 50000,
    step_factor: float = 2.0
) -> None:
    """
    Run comprehensive scaling benchmark across dataset sizes.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save results
    max_samples : int
        Maximum number of samples to test
    step_factor : float
        Multiplicative factor between sample sizes
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    suite = PerformanceSuite()
    
    # Generate sample sizes
    sample_sizes = []
    n = 1000
    while n <= max_samples:
        sample_sizes.append(n)
        n = int(n * step_factor)
    
    logger.info(f"Running scaling benchmark on sample sizes: {sample_sizes}")
    
    # Test both GEX and TCR scaling
    for data_type in ['gex', 'tcr']:
        logger.info(f"Running {data_type} scaling benchmark...")
        
        scaling_results = suite.run_scaling_benchmark(
            data_type=data_type,
            sample_sizes=sample_sizes,
            nbr_fracs=[0.01, 0.05]
        )
        
        # Save scaling results
        for result in scaling_results:
            df = pd.DataFrame({
                'sample_size': result.sample_sizes,
                'query_time': result.query_times,
                'memory_mb': result.memory_usage,
                'speedup_vs_sklearn': result.speedup_vs_sklearn
            })
            
            filename = f"{data_type}_scaling_{result.backend}.csv"
            df.to_csv(output_path / filename, index=False)
            logger.info(f"Saved {filename}")
    
    # Generate overall report
    suite.save_results(output_path / "benchmark_summary")
    
    logger.info(f"Comprehensive benchmark completed. Results in {output_path}")


def validate_faiss_accuracy(
    n_samples: int = 5000,
    tolerance: float = 0.95
) -> bool:
    """
    Validate that FAISS produces sufficiently accurate results vs sklearn.
    
    This is a compatibility wrapper around the comprehensive accuracy validation
    suite. For detailed validation, use conga.accuracy_validation directly.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to test
    tolerance : float
        Minimum required accuracy (fraction of identical neighbors)
        
    Returns:
    --------
    bool
        True if all FAISS backends meet accuracy requirements
    """
    try:
        from .accuracy_validation import quick_accuracy_check
        return quick_accuracy_check(n_samples=n_samples, tolerance=tolerance)
    except ImportError:
        logger.warning("accuracy_validation module not available, using legacy validation")
        return _legacy_validate_faiss_accuracy(n_samples, tolerance)

def _legacy_validate_faiss_accuracy(
    n_samples: int = 5000,
    tolerance: float = 0.95
) -> bool:
    """Legacy FAISS accuracy validation (fallback)."""
    suite = PerformanceSuite()
    
    # Test both data types
    for data_type in ['gex', 'tcr']:
        logger.info(f"Validating FAISS accuracy for {data_type} data...")
        
        # Generate test data
        if data_type == 'gex':
            X = DatasetGenerator.generate_gex_data(n_samples)
        else:
            X = DatasetGenerator.generate_tcr_vector_data(n_samples)
        
        exclude_groups = DatasetGenerator.generate_exclude_groups(n_samples)
        
        # Run benchmark
        results = suite.benchmark_single_dataset(
            X=X,
            data_type=data_type,
            nbr_fracs=[0.01, 0.05],
            exclude_groups=exclude_groups,
            validate_accuracy=True
        )
        
        # Check accuracy for FAISS backends
        for result in results:
            if result.backend.startswith('faiss') and result.accuracy_vs_baseline is not None:
                if result.accuracy_vs_baseline < tolerance:
                    logger.error(
                        f"{result.backend} accuracy {result.accuracy_vs_baseline:.3f} "
                        f"< {tolerance} for {data_type} data"
                    )
                    return False
                else:
                    logger.info(
                        f"{result.backend} accuracy {result.accuracy_vs_baseline:.3f} "
                        f">= {tolerance} for {data_type} data ✓"
                    )
    
    logger.info("FAISS accuracy validation passed ✓")
    return True


@dataclass
class PerformanceTestConfig:
    """Configuration for comprehensive performance testing."""
    sample_sizes: List[int]
    data_types: List[str]  # ['gex', 'tcr']
    nbr_fracs: List[float]
    n_iterations: int = 3
    warmup_iterations: int = 1
    test_backends: Optional[List[str]] = None
    max_memory_gb: float = 16.0
    output_dir: str = "benchmark_results"
    include_accuracy: bool = True
    include_memory_profiling: bool = True
    include_gpu_profiling: bool = True

@dataclass
class HardwareProfile:
    """Hardware configuration profile for testing."""
    cpu_cores: int
    memory_gb: float
    gpu_available: bool
    gpu_memory_gb: float
    faiss_gpu_available: bool
    faiss_cpu_available: bool
    platform: str

class ComprehensivePerformanceSuite:
    """
    Comprehensive performance testing suite for FAISS acceleration.
    
    Extends the base PerformanceSuite with advanced capabilities:
    - Multi-iteration statistical testing
    - Hardware configuration profiling
    - Detailed memory and GPU usage tracking
    - Performance scaling analysis and predictions
    - Comprehensive report generation with visualizations
    """
    
    def __init__(self, config: PerformanceTestConfig):
        self.config = config
        self.base_suite = PerformanceSuite(max_memory_gb=config.max_memory_gb)
        self.hardware_profile = self._detect_hardware()
        self.detailed_results = []
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Comprehensive performance suite initialized")
        logger.info(f"Hardware: {self.hardware_profile}")

    def _detect_hardware(self) -> HardwareProfile:
        """Detect hardware configuration for testing context."""
        import platform
        
        # CPU information
        cpu_cores = psutil.cpu_count(logical=True)
        
        # Memory information
        memory_info = psutil.virtual_memory()
        memory_gb = memory_info.total / (1024**3)
        
        # GPU information
        gpu_available = False
        gpu_memory_gb = 0.0
        
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                gpu_available = True
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_gb = gpu_memory_info.total / (1024**3)
        except (ImportError, Exception):
            pass
        
        # FAISS availability
        backend_info = get_backend_info()
        
        return HardwareProfile(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            gpu_available=gpu_available,
            gpu_memory_gb=gpu_memory_gb,
            faiss_gpu_available=backend_info['faiss_gpu_available'],
            faiss_cpu_available=backend_info['faiss_cpu_available'],
            platform=platform.system()
        )

    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """
        Run the complete comprehensive performance test suite.
        
        Returns:
        --------
        Dict[str, Any]
            Comprehensive test results with scaling analysis, recommendations, etc.
        """
        logger.info("=== Starting Comprehensive FAISS Performance Test Suite ===")
        
        # Initialize results storage
        all_results = {
            'hardware_profile': self.hardware_profile,
            'config': self.config,
            'scaling_results': {},
            'detailed_benchmarks': [],
            'statistical_analysis': {},
            'performance_recommendations': {},
            'accuracy_validation': {},
            'memory_analysis': {}
        }
        
        # 1. Hardware validation and backend availability
        logger.info("Step 1: Validating hardware and backend availability...")
        backend_validation = self._validate_backends()
        all_results['backend_validation'] = backend_validation
        
        if not any(backend_validation.values()):
            logger.error("No FAISS backends available - testing sklearn only")
        
        # 2. Multi-iteration performance testing
        logger.info("Step 2: Running multi-iteration performance tests...")
        for data_type in self.config.data_types:
            for n_samples in self.config.sample_sizes:
                iteration_results = self._run_multi_iteration_test(data_type, n_samples)
                all_results['detailed_benchmarks'].append(iteration_results)
        
        # 3. Scaling analysis
        logger.info("Step 3: Analyzing performance scaling...")
        scaling_results = self._analyze_scaling_behavior()
        all_results['scaling_results'] = scaling_results
        
        # 4. Memory analysis
        if self.config.include_memory_profiling:
            logger.info("Step 4: Running memory profiling analysis...")
            memory_analysis = self._analyze_memory_scaling()
            all_results['memory_analysis'] = memory_analysis
        
        # 5. Accuracy validation
        if self.config.include_accuracy:
            logger.info("Step 5: Running accuracy validation...")
            accuracy_results = self._validate_accuracy_comprehensive()
            all_results['accuracy_validation'] = accuracy_results
        
        # 6. Generate recommendations
        logger.info("Step 6: Generating performance recommendations...")
        recommendations = self._generate_comprehensive_recommendations(all_results)
        all_results['performance_recommendations'] = recommendations
        
        # 7. Save comprehensive results
        self._save_comprehensive_results(all_results)
        
        logger.info("=== Comprehensive Performance Test Suite Completed ===")
        return all_results
    
    def _validate_backends(self) -> Dict[str, bool]:
        """Validate available backends and their functionality."""
        validation = {}
        
        # Test sklearn (always available)
        validation['sklearn'] = True
        logger.info("✓ sklearn backend available")
        
        # Test FAISS-CPU
        if self.hardware_profile.faiss_cpu_available:
            try:
                # Quick validation test
                X = DatasetGenerator.generate_gex_data(100, 50)
                searcher = FaissNeighborSearcher(force_backend=Backend.FAISS_CPU)
                result = searcher.search_neighbors(X, [0.1], data_type='test')
                validation['faiss_cpu'] = len(result.neighbors) > 0
                logger.info("✓ FAISS-CPU backend validated")
            except Exception as e:
                validation['faiss_cpu'] = False
                logger.warning(f"✗ FAISS-CPU validation failed: {e}")
        else:
            validation['faiss_cpu'] = False
            logger.info("✗ FAISS-CPU not available")
        
        # Test FAISS-GPU
        if self.hardware_profile.faiss_gpu_available:
            try:
                X = DatasetGenerator.generate_gex_data(100, 50)
                searcher = FaissNeighborSearcher(force_backend=Backend.FAISS_GPU)
                result = searcher.search_neighbors(X, [0.1], data_type='test')
                validation['faiss_gpu'] = len(result.neighbors) > 0
                logger.info("✓ FAISS-GPU backend validated")
            except Exception as e:
                validation['faiss_gpu'] = False
                logger.warning(f"✗ FAISS-GPU validation failed: {e}")
        else:
            validation['faiss_gpu'] = False
            logger.info("✗ FAISS-GPU not available")
        
        return validation
    
    def _run_multi_iteration_test(self, data_type: str, n_samples: int) -> Dict[str, Any]:
        """Run multi-iteration statistical test for a specific configuration."""
        logger.info(f"Running {self.config.n_iterations} iterations: {data_type} data, {n_samples} samples")
        
        # Generate test data once
        if data_type == 'gex':
            X = DatasetGenerator.generate_gex_data(n_samples, 2000)
            n_features = 2000
        elif data_type == 'tcr':
            X = DatasetGenerator.generate_tcr_vector_data(n_samples, 1136)
            n_features = 1136
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        exclude_groups = DatasetGenerator.generate_exclude_groups(n_samples)
        
        # Determine backends to test
        available_backends = []
        if self.hardware_profile.faiss_gpu_available:
            available_backends.append('faiss-gpu')
        if self.hardware_profile.faiss_cpu_available:
            available_backends.append('faiss-cpu')
        available_backends.append('sklearn')
        
        if self.config.test_backends:
            available_backends = [b for b in available_backends if b in self.config.test_backends]
        
        iteration_results = {
            'data_type': data_type,
            'n_samples': n_samples,
            'n_features': n_features,
            'backends': {}
        }
        
        # Test each backend with multiple iterations
        for backend in available_backends:
            backend_results = {
                'query_times': [],
                'index_times': [],
                'memory_usage': [],
                'accuracy_scores': []
            }
            
            logger.debug(f"  Testing {backend} backend...")
            
            # Warmup iterations
            for _ in range(self.config.warmup_iterations):
                try:
                    self.base_suite.benchmark_single_dataset(
                        X=X, data_type=data_type, nbr_fracs=self.config.nbr_fracs,
                        exclude_groups=exclude_groups, test_backends=[backend],
                        validate_accuracy=False
                    )
                except Exception as e:
                    logger.warning(f"Warmup iteration failed for {backend}: {e}")
            
            # Actual test iterations
            for iteration in range(self.config.n_iterations):
                try:
                    results = self.base_suite.benchmark_single_dataset(
                        X=X, data_type=data_type, nbr_fracs=self.config.nbr_fracs,
                        exclude_groups=exclude_groups, test_backends=[backend],
                        validate_accuracy=self.config.include_accuracy
                    )
                    
                    if results:
                        result = results[0]  # Should only be one result
                        backend_results['query_times'].append(result.query_time)
                        backend_results['index_times'].append(result.index_time)
                        backend_results['memory_usage'].append(result.peak_memory_mb)
                        
                        if result.accuracy_vs_baseline is not None:
                            backend_results['accuracy_scores'].append(result.accuracy_vs_baseline)
                        
                except Exception as e:
                    logger.warning(f"Iteration {iteration+1} failed for {backend}: {e}")
            
            # Calculate statistics
            if backend_results['query_times']:
                backend_results['query_time_mean'] = np.mean(backend_results['query_times'])
                backend_results['query_time_std'] = np.std(backend_results['query_times'])
                backend_results['query_time_min'] = np.min(backend_results['query_times'])
                backend_results['query_time_max'] = np.max(backend_results['query_times'])
                
                backend_results['memory_mean'] = np.mean(backend_results['memory_usage'])
                backend_results['memory_std'] = np.std(backend_results['memory_usage'])
                
                if backend_results['accuracy_scores']:
                    backend_results['accuracy_mean'] = np.mean(backend_results['accuracy_scores'])
                    backend_results['accuracy_std'] = np.std(backend_results['accuracy_scores'])
                
                logger.info(f"    {backend}: {backend_results['query_time_mean']:.3f}±{backend_results['query_time_std']:.3f}s, "
                           f"{backend_results['memory_mean']:.1f}±{backend_results['memory_std']:.1f}MB")
            
            iteration_results['backends'][backend] = backend_results
        
        return iteration_results

    def _analyze_scaling_behavior(self) -> Dict[str, Any]:
        """Analyze performance scaling across dataset sizes."""
        scaling_results = {}
        
        for data_type in self.config.data_types:
            scaling_results[data_type] = {}
            
            # Extract scaling data from detailed benchmarks  
            size_data = {}
            for result in self.detailed_results:
                if result['data_type'] == data_type:
                    n_samples = result['n_samples']
                    size_data[n_samples] = result['backends']
            
            # Analyze each backend's scaling
            all_backends = set()
            for size_result in size_data.values():
                all_backends.update(size_result.keys())
            
            for backend in all_backends:
                backend_scaling = {
                    'sample_sizes': [],
                    'query_times': [],
                    'memory_usage': [],
                    'speedup_vs_sklearn': []
                }
                
                sklearn_times = {}
                
                # Collect data points
                for n_samples in sorted(size_data.keys()):
                    if backend in size_data[n_samples]:
                        backend_data = size_data[n_samples][backend]
                        if 'query_time_mean' in backend_data:
                            backend_scaling['sample_sizes'].append(n_samples)
                            backend_scaling['query_times'].append(backend_data['query_time_mean'])
                            backend_scaling['memory_usage'].append(backend_data['memory_mean'])
                            
                            # Store sklearn baseline for speedup calculation
                            if backend == 'sklearn':
                                sklearn_times[n_samples] = backend_data['query_time_mean']
                
                # Calculate speedups vs sklearn
                for i, n_samples in enumerate(backend_scaling['sample_sizes']):
                    if n_samples in sklearn_times and sklearn_times[n_samples] > 0:
                        speedup = sklearn_times[n_samples] / backend_scaling['query_times'][i]
                        backend_scaling['speedup_vs_sklearn'].append(speedup)
                    else:
                        backend_scaling['speedup_vs_sklearn'].append(1.0)
                
                # Fit scaling curves if enough data points
                if len(backend_scaling['sample_sizes']) >= 3:
                    backend_scaling['scaling_analysis'] = self._fit_scaling_curves(
                        backend_scaling['sample_sizes'],
                        backend_scaling['query_times'],
                        backend_scaling['memory_usage']
                    )
                
                scaling_results[data_type][backend] = backend_scaling
        
        return scaling_results

    def _fit_scaling_curves(self, sizes: List[int], times: List[float], 
                           memory: List[float]) -> Dict[str, Any]:
        """Fit scaling curves to performance data."""
        try:
            sizes_arr = np.array(sizes)
            times_arr = np.array(times)
            memory_arr = np.array(memory)
            
            # Fit polynomial curves (linear, quadratic)
            time_linear_fit = np.polyfit(np.log(sizes_arr), np.log(times_arr), 1)
            memory_linear_fit = np.polyfit(np.log(sizes_arr), np.log(memory_arr), 1)
            
            # Calculate R² scores
            time_pred = np.exp(np.polyval(time_linear_fit, np.log(sizes_arr)))
            time_r2 = 1 - np.sum((times_arr - time_pred)**2) / np.sum((times_arr - np.mean(times_arr))**2)
            
            memory_pred = np.exp(np.polyval(memory_linear_fit, np.log(sizes_arr)))
            memory_r2 = 1 - np.sum((memory_arr - memory_pred)**2) / np.sum((memory_arr - np.mean(memory_arr))**2)
            
            return {
                'time_complexity_exponent': time_linear_fit[0],
                'time_r2': time_r2,
                'memory_complexity_exponent': memory_linear_fit[0],
                'memory_r2': memory_r2,
                'projected_performance': self._project_performance(sizes_arr, times_arr, time_linear_fit)
            }
            
        except Exception as e:
            logger.warning(f"Scaling curve fitting failed: {e}")
            return {'error': str(e)}

    def _project_performance(self, sizes: np.ndarray, times: np.ndarray, 
                           fit_params: np.ndarray) -> Dict[int, float]:
        """Project performance to larger dataset sizes."""
        projections = {}
        
        # Project to common benchmark sizes
        target_sizes = [50000, 100000, 200000, 500000, 1000000]
        
        for target_size in target_sizes:
            if target_size > max(sizes):
                projected_time = np.exp(np.polyval(fit_params, np.log(target_size)))
                projections[target_size] = projected_time
        
        return projections

    def _analyze_memory_scaling(self) -> Dict[str, Any]:
        """Analyze memory usage patterns and scaling."""
        memory_analysis = {}
        
        for data_type in self.config.data_types:
            memory_analysis[data_type] = {}
            
            # Extract memory data
            for result in self.detailed_results:
                if result['data_type'] == data_type:
                    n_samples = result['n_samples']
                    
                    for backend, backend_data in result['backends'].items():
                        if backend not in memory_analysis[data_type]:
                            memory_analysis[data_type][backend] = {
                                'sample_sizes': [],
                                'peak_memory': [],
                                'memory_efficiency': []
                            }
                        
                        if 'memory_mean' in backend_data:
                            memory_analysis[data_type][backend]['sample_sizes'].append(n_samples)
                            memory_analysis[data_type][backend]['peak_memory'].append(backend_data['memory_mean'])
                            
                            # Calculate memory efficiency (MB per 1k samples)
                            efficiency = backend_data['memory_mean'] / (n_samples / 1000)
                            memory_analysis[data_type][backend]['memory_efficiency'].append(efficiency)
            
            # Analyze memory scaling for each backend
            for backend in memory_analysis[data_type]:
                backend_data = memory_analysis[data_type][backend]
                if len(backend_data['sample_sizes']) >= 2:
                    # Calculate memory growth rate
                    sizes = np.array(backend_data['sample_sizes'])
                    memory = np.array(backend_data['peak_memory'])
                    
                    if len(sizes) >= 3:
                        # Fit log-log curve to determine scaling
                        try:
                            fit = np.polyfit(np.log(sizes), np.log(memory), 1)
                            backend_data['memory_scaling_exponent'] = fit[0]
                            
                            # Classify scaling behavior
                            if fit[0] < 1.2:
                                backend_data['scaling_classification'] = 'sub-linear'
                            elif fit[0] < 1.8:
                                backend_data['scaling_classification'] = 'linear'
                            elif fit[0] < 2.2:
                                backend_data['scaling_classification'] = 'quadratic'
                            else:
                                backend_data['scaling_classification'] = 'super-quadratic'
                                
                        except Exception as e:
                            logger.warning(f"Memory scaling analysis failed for {backend}: {e}")
        
        return memory_analysis

    def _validate_accuracy_comprehensive(self) -> Dict[str, Any]:
        """Run comprehensive accuracy validation."""
        accuracy_results = {}
        
        # Use moderate size for accuracy testing to ensure reasonable runtime
        test_size = min(5000, max(self.config.sample_sizes))
        
        for data_type in self.config.data_types:
            logger.info(f"Validating accuracy for {data_type} data...")
            
            # Generate test data
            if data_type == 'gex':
                X = DatasetGenerator.generate_gex_data(test_size, 2000)
            else:
                X = DatasetGenerator.generate_tcr_vector_data(test_size, 1136)
            
            exclude_groups = DatasetGenerator.generate_exclude_groups(test_size)
            
            # Run accuracy validation
            results = self.base_suite.benchmark_single_dataset(
                X=X, data_type=data_type, nbr_fracs=[0.01, 0.05, 0.10],
                exclude_groups=exclude_groups, validate_accuracy=True
            )
            
            backend_accuracy = {}
            for result in results:
                if result.accuracy_vs_baseline is not None:
                    backend_accuracy[result.backend] = {
                        'accuracy': result.accuracy_vs_baseline,
                        'passed': result.accuracy_vs_baseline >= 0.95  # 95% threshold
                    }
            
            accuracy_results[data_type] = backend_accuracy
        
        return accuracy_results

    def _generate_comprehensive_recommendations(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive performance recommendations."""
        recommendations = {
            'hardware_specific': {},
            'dataset_size_specific': {},
            'backend_selection': {},
            'optimization_suggestions': []
        }
        
        # Hardware-specific recommendations
        hw = self.hardware_profile
        if hw.faiss_gpu_available and hw.gpu_memory_gb >= 8:
            recommendations['hardware_specific']['gpu'] = "FAISS-GPU recommended for large datasets with high-end GPU"
        elif hw.faiss_gpu_available:
            recommendations['hardware_specific']['gpu'] = "FAISS-GPU available but limited GPU memory may cause fallbacks"
        else:
            recommendations['hardware_specific']['gpu'] = "No FAISS-GPU available, use CPU backends"
        
        if hw.faiss_cpu_available:
            recommendations['hardware_specific']['cpu'] = f"FAISS-CPU available with {hw.cpu_cores} cores"
        
        # Dataset size specific recommendations
        scaling_results = all_results.get('scaling_results', {})
        for data_type, backend_scaling in scaling_results.items():
            recommendations['dataset_size_specific'][data_type] = {}
            
            # Find crossover points where FAISS becomes beneficial
            for backend in ['faiss-gpu', 'faiss-cpu']:
                if backend in backend_scaling:
                    speedups = backend_scaling[backend].get('speedup_vs_sklearn', [])
                    sizes = backend_scaling[backend].get('sample_sizes', [])
                    
                    # Find where speedup > 2x consistently
                    beneficial_size = None
                    for i, (size, speedup) in enumerate(zip(sizes, speedups)):
                        if speedup >= 2.0:
                            beneficial_size = size
                            break
                    
                    if beneficial_size:
                        recommendations['dataset_size_specific'][data_type][backend] = \
                            f"Recommended for datasets >= {beneficial_size:,} samples"
                    else:
                        recommendations['dataset_size_specific'][data_type][backend] = \
                            "Marginal benefit observed in tested range"
        
        # Overall backend selection strategy
        recommendations['backend_selection'] = {
            'small_datasets': "sklearn sufficient for < 5k samples",
            'medium_datasets': "FAISS-CPU recommended for 5k-50k samples",
            'large_datasets': "FAISS-GPU preferred for > 50k samples (if available)",
            'fallback_strategy': "Automatic backend selection with graceful fallback"
        }
        
        # Optimization suggestions
        accuracy_results = all_results.get('accuracy_validation', {})
        for data_type, backend_results in accuracy_results.items():
            for backend, result in backend_results.items():
                if not result.get('passed', True):
                    recommendations['optimization_suggestions'].append(
                        f"Accuracy concern: {backend} on {data_type} data below 95% threshold"
                    )
        
        memory_analysis = all_results.get('memory_analysis', {})
        for data_type, backend_memory in memory_analysis.items():
            for backend, memory_data in backend_memory.items():
                scaling_class = memory_data.get('scaling_classification')
                if scaling_class == 'super-quadratic':
                    recommendations['optimization_suggestions'].append(
                        f"Memory concern: {backend} on {data_type} has super-quadratic scaling"
                    )
        
        return recommendations

    def _save_comprehensive_results(self, results: Dict[str, Any]):
        """Save comprehensive results to files."""
        output_path = Path(self.config.output_dir)
        
        # Save detailed results as JSON
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        def recursive_convert(d):
            if isinstance(d, dict):
                return {k: recursive_convert(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [recursive_convert(v) for v in d]
            elif hasattr(d, '__dict__'):
                # Handle dataclass and other objects with __dict__
                return recursive_convert(d.__dict__)
            else:
                return convert_numpy(d)
        
        results_json = recursive_convert(results)
        
        with open(output_path / "comprehensive_results.json", 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # Save summary report as text
        with open(output_path / "performance_summary.txt", 'w') as f:
            f.write("FAISS Performance Test Suite - Comprehensive Results\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Hardware Configuration:\n")
            f.write(f"  CPU Cores: {self.hardware_profile.cpu_cores}\n")
            f.write(f"  Memory: {self.hardware_profile.memory_gb:.1f} GB\n")
            f.write(f"  GPU Available: {self.hardware_profile.gpu_available}\n")
            f.write(f"  FAISS-GPU: {self.hardware_profile.faiss_gpu_available}\n")
            f.write(f"  FAISS-CPU: {self.hardware_profile.faiss_cpu_available}\n\n")
            
            f.write("Performance Recommendations:\n")
            for category, recs in results['performance_recommendations'].items():
                f.write(f"\n{category.replace('_', ' ').title()}:\n")
                if isinstance(recs, dict):
                    for key, value in recs.items():
                        f.write(f"  {key}: {value}\n")
                elif isinstance(recs, list):
                    for item in recs:
                        f.write(f"  - {item}\n")
                else:
                    f.write(f"  {recs}\n")
        
        logger.info(f"Comprehensive results saved to {output_path}")


# Factory function for creating comprehensive test configurations
def create_comprehensive_test_config(
    test_scale: str = "standard",
    data_types: Optional[List[str]] = None,
    output_dir: str = "benchmark_results"
) -> PerformanceTestConfig:
    """
    Create a PerformanceTestConfig for different testing scales.
    
    Parameters:
    -----------
    test_scale : str
        Scale of testing: 'quick', 'standard', 'comprehensive', or 'stress'
    data_types : Optional[List[str]]
        Data types to test (default: both 'gex' and 'tcr')
    output_dir : str
        Directory for results
        
    Returns:
    --------
    PerformanceTestConfig
        Configured test parameters
    """
    if data_types is None:
        data_types = ['gex', 'tcr']
    
    configs = {
        'quick': PerformanceTestConfig(
            sample_sizes=[1000, 5000, 10000],
            data_types=data_types,
            nbr_fracs=[0.01, 0.05],
            n_iterations=2,
            warmup_iterations=1,
            max_memory_gb=8.0,
            output_dir=output_dir
        ),
        'standard': PerformanceTestConfig(
            sample_sizes=[1000, 5000, 10000, 20000, 50000],
            data_types=data_types,
            nbr_fracs=[0.01, 0.05, 0.10],
            n_iterations=3,
            warmup_iterations=1,
            max_memory_gb=16.0,
            output_dir=output_dir
        ),
        'comprehensive': PerformanceTestConfig(
            sample_sizes=[1000, 2000, 5000, 10000, 20000, 50000, 100000],
            data_types=data_types,
            nbr_fracs=[0.01, 0.02, 0.05, 0.10],
            n_iterations=5,
            warmup_iterations=2,
            max_memory_gb=32.0,
            output_dir=output_dir
        ),
        'stress': PerformanceTestConfig(
            sample_sizes=[1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000],
            data_types=data_types,
            nbr_fracs=[0.01, 0.02, 0.05, 0.10, 0.20],
            n_iterations=5,
            warmup_iterations=2,
            max_memory_gb=64.0,
            output_dir=output_dir
        )
    }
    
    if test_scale not in configs:
        raise ValueError(f"Unknown test scale '{test_scale}'. Choose from: {list(configs.keys())}")
    
    return configs[test_scale]


if __name__ == "__main__":
    # Example usage and testing
    
    # Quick benchmarks (existing functionality)
    print("Running quick GEX benchmark...")
    gex_results = quick_gex_benchmark(n_samples=5000)
    print(gex_results)
    
    print("\nRunning quick TCR benchmark...")
    tcr_results = quick_tcr_benchmark(n_samples=3000)
    print(tcr_results)
    
    # Accuracy validation
    print("\nValidating FAISS accuracy...")
    accuracy_ok = validate_faiss_accuracy()
    print(f"Accuracy validation: {'PASS' if accuracy_ok else 'FAIL'}")
    
    # Comprehensive performance test suite
    print("\n" + "="*60)
    print("COMPREHENSIVE PERFORMANCE TEST SUITE")
    print("="*60)
    
    # Create test configuration
    config = create_comprehensive_test_config(test_scale="quick", output_dir="benchmark_results")
    
    # Run comprehensive suite
    suite = ComprehensivePerformanceSuite(config)
    comprehensive_results = suite.run_comprehensive_test_suite()
    
    # Print summary
    print("\n=== COMPREHENSIVE TEST RESULTS SUMMARY ===")
    print(f"Hardware: {comprehensive_results['hardware_profile']}")
    
    recommendations = comprehensive_results['performance_recommendations']
    print(f"\nRecommendations:")
    for category, recs in recommendations.items():
        print(f"  {category}: {recs}")
    
    print(f"\nDetailed results saved to: {config.output_dir}/")
    print("Files generated:")
    print("  - comprehensive_results.json (detailed data)")
    print("  - performance_summary.txt (summary report)")
    
    # Demonstrate different test scales
    print("\n" + "="*50)
    print("AVAILABLE TEST CONFIGURATIONS")
    print("="*50)
    
    for scale in ['quick', 'standard', 'comprehensive', 'stress']:
        test_config = create_comprehensive_test_config(test_scale=scale)
        print(f"\n{scale.upper()} test:")
        print(f"  Sample sizes: {test_config.sample_sizes}")
        print(f"  Iterations: {test_config.n_iterations}")
        print(f"  Memory limit: {test_config.max_memory_gb} GB")
        print(f"  Estimated runtime: {len(test_config.sample_sizes) * len(test_config.data_types) * test_config.n_iterations * 30:.0f}-{len(test_config.sample_sizes) * len(test_config.data_types) * test_config.n_iterations * 120:.0f} seconds")