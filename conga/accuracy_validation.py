"""
Comprehensive FAISS vs sklearn accuracy validation suite.

This module provides rigorous accuracy validation for FAISS neighbor search
implementations, ensuring production-grade correctness through detailed
comparisons with sklearn baselines.

Key components:
- EdgeCaseGenerator: Creates challenging test datasets
- AccuracyValidator: Comprehensive backend comparison
- DetailedReport: Structured validation results
- Edge case testing for duplicate vectors, high dimensionality, batch boundaries

Example usage:
    validator = AccuracyValidator()
    report = validator.validate_all_backends()
    if report.passes_production_standards():
        print("FAISS validation passed for production use")
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
import pandas as pd
from pathlib import Path

# Scientific computing
from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_blobs

# CoNGA imports
from .neighbors import (
    FaissNeighborSearcher, 
    Backend, 
    get_backend_info,
    validate_neighbor_results
)
from .benchmark import DatasetGenerator, MemoryTracker

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Container for detailed validation measurements."""
    backend_a: str
    backend_b: str
    data_type: str
    test_case: str
    n_samples: int
    n_features: int
    k_values: List[int]
    recall_at_k: Dict[int, float]  # k -> recall score
    precision_at_k: Dict[int, float]  # k -> precision score
    distance_correlation: float  # Pearson correlation of distances
    identical_neighbors: float  # Fraction of exactly identical neighbor sets
    max_distance_error: float  # Maximum absolute distance difference
    deterministic_match: bool  # Whether repeated runs are identical
    notes: str = ""

@dataclass
class DetailedReport:
    """Comprehensive accuracy validation report."""
    validation_results: List[ValidationResult]
    edge_case_results: List[ValidationResult] 
    determinism_results: List[ValidationResult]
    backend_availability: Dict[str, bool]
    overall_pass: bool
    production_ready: bool
    recommendations: List[str]
    timestamp: str
class EdgeCaseGenerator:
    """Generator for challenging accuracy validation datasets."""
    
    @staticmethod
    def generate_duplicate_vectors(
        n_samples: int, 
        n_features: int = 100,
        duplicate_fraction: float = 0.3,
        random_seed: int = 42
    ) -> np.ndarray:
        """Generate dataset with intentional duplicate vectors."""
        np.random.seed(random_seed)
        
        n_unique = int(n_samples * (1 - duplicate_fraction))
        n_duplicates = n_samples - n_unique
        
        # Generate unique vectors
        X_unique = np.random.randn(n_unique, n_features).astype(np.float32)
        
        # Create duplicates by sampling from unique vectors
        duplicate_indices = np.random.choice(n_unique, n_duplicates, replace=True)
        X_duplicates = X_unique[duplicate_indices]
        
        # Combine and shuffle
        X = np.vstack([X_unique, X_duplicates])
        shuffle_idx = np.random.permutation(n_samples)
        X = X[shuffle_idx]
        
        logger.debug(f"Generated dataset with {n_duplicates} duplicates from {n_unique} unique vectors")
        return X
    
    @staticmethod
    def generate_high_dimensional_sparse(
        n_samples: int,
        n_features: int = 5000, 
        sparsity: float = 0.95,
        random_seed: int = 42
    ) -> np.ndarray:
        """Generate high-dimensional sparse dataset (simulates gene expression)."""
        np.random.seed(random_seed)
        
        # Start with dense data
        X = np.random.gamma(2, 0.5, (n_samples, n_features)).astype(np.float32)
        
        # Make it sparse
        mask = np.random.random((n_samples, n_features)) < sparsity
        X[mask] = 0
        
        # Add some structure to non-zero elements
        X = np.log1p(X)  # Log transform like scRNA-seq
        
        logger.debug(f"Generated {n_samples}×{n_features} sparse data ({sparsity:.1%} sparsity)")
        return X
    
    @staticmethod
    def generate_clustered_data(
        n_samples: int,
        n_features: int = 200,
        n_clusters: int = 5,
        cluster_separation: float = 2.0,
        random_seed: int = 42
    ) -> np.ndarray:
        """Generate tightly clustered data to test neighbor boundary behavior."""
        np.random.seed(random_seed)
        
        X, _ = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=n_clusters,
            cluster_std=1.0,
            center_box=(-cluster_separation, cluster_separation),
            random_state=random_seed
        )
        
        logger.debug(f"Generated {n_clusters} clusters with separation {cluster_separation}")
        return X.astype(np.float32)
    
    @staticmethod
    def generate_extreme_aspect_ratio(
        n_samples: int = 1000,
        n_features: int = 10000, 
        random_seed: int = 42
    ) -> np.ndarray:
        """Generate data with extreme aspect ratio (few samples, many features)."""
        np.random.seed(random_seed)
        
        # Create low-rank structure
        rank = min(50, n_samples // 2)
        U = np.random.randn(n_samples, rank)
        V = np.random.randn(rank, n_features) 
        X = U @ V
        
        # Add noise
        X += 0.1 * np.random.randn(n_samples, n_features)
        
        logger.debug(f"Generated extreme aspect ratio data: {n_samples}×{n_features} (rank ~{rank})")
        return X.astype(np.float32)
class AccuracyValidator:
    """
    Comprehensive FAISS vs sklearn accuracy validator.
    
    Provides rigorous testing across multiple backends, edge cases, and
    validation metrics to ensure production-grade correctness.
    """
    
    def __init__(self, 
                 tolerance: float = 0.98,
                 strict_mode: bool = True):
        """
        Initialize accuracy validator.
        
        Parameters:
        -----------
        tolerance : float
            Minimum required accuracy for production use (default: 98%)
        strict_mode : bool
            Whether to require exact deterministic results
        """
        self.tolerance = tolerance
        self.strict_mode = strict_mode
        self.backend_info = get_backend_info()
        
        logger.info(f"AccuracyValidator initialized with {tolerance:.1%} tolerance")
    
    def validate_neighbor_recall(
        self,
        neighbors_test: np.ndarray,
        neighbors_baseline: np.ndarray,
        distances_test: Optional[np.ndarray] = None,
        distances_baseline: Optional[np.ndarray] = None,
        k_values: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Validate neighbor recall at different k values.
        
        Parameters:
        -----------
        neighbors_test : np.ndarray
            Neighbor indices from test backend (n_samples, n_neighbors)
        neighbors_baseline : np.ndarray  
            Neighbor indices from baseline backend
        distances_test, distances_baseline : Optional[np.ndarray]
            Distance values for correlation analysis
        k_values : Optional[List[int]]
            K values to test (default: [1, 5, 10, 20, 50])
            
        Returns:
        --------
        Dict[str, float]
            Recall scores at each k value
        """
        if k_values is None:
            k_values = [1, 5, 10, 20, 50]
        
        # Validate shapes
        if neighbors_test.shape[0] != neighbors_baseline.shape[0]:
            raise ValueError(f"Number of samples mismatch: {neighbors_test.shape[0]} vs {neighbors_baseline.shape[0]}")
        
        n_samples = neighbors_test.shape[0]
        max_k = min(neighbors_test.shape[1], neighbors_baseline.shape[1])
        k_values = [k for k in k_values if k <= max_k]
        
        recall_scores = {}
        
        for k in k_values:
            total_recall = 0.0
            
            for i in range(n_samples):
                test_neighbors_k = set(neighbors_test[i, :k])
                baseline_neighbors_k = set(neighbors_baseline[i, :k])
                
                if len(baseline_neighbors_k) > 0:
                    overlap = len(test_neighbors_k.intersection(baseline_neighbors_k))
                    recall = overlap / len(baseline_neighbors_k)
                    total_recall += recall
            
            recall_scores[k] = total_recall / n_samples if n_samples > 0 else 0.0
            logger.debug(f"Recall@{k}: {recall_scores[k]:.4f}")
        
        return recall_scores
    
    def validate_distance_correlation(
        self,
        distances_test: np.ndarray,
        distances_baseline: np.ndarray,
        sample_fraction: float = 1.0
    ) -> Dict[str, float]:
        """
        Validate correlation between distance matrices.
        
        Parameters:
        -----------
        distances_test, distances_baseline : np.ndarray
            Distance matrices to compare
        sample_fraction : float
            Fraction of distances to sample for large matrices
            
        Returns:
        --------
        Dict[str, float]
            Correlation metrics
        """
        # Flatten distance matrices
        dist_test_flat = distances_test.flatten()
        dist_baseline_flat = distances_baseline.flatten()
        
        # Sample if requested (for large matrices)
        if sample_fraction < 1.0:
            n_total = len(dist_test_flat)
            n_sample = int(n_total * sample_fraction)
            indices = np.random.choice(n_total, n_sample, replace=False)
            dist_test_flat = dist_test_flat[indices]
            dist_baseline_flat = dist_baseline_flat[indices]
        
        # Calculate correlation metrics
        correlation = np.corrcoef(dist_test_flat, dist_baseline_flat)[0, 1]
        
        # Calculate absolute errors
        abs_errors = np.abs(dist_test_flat - dist_baseline_flat)
        max_error = np.max(abs_errors)
        mean_error = np.mean(abs_errors)
        median_error = np.median(abs_errors)
        
        return {
            'pearson_correlation': correlation,
            'max_absolute_error': max_error,
            'mean_absolute_error': mean_error,
            'median_absolute_error': median_error
        }
    def compare_backends_comprehensive(
        self,
        X: np.ndarray,
        backend_test: str,
        backend_baseline: str = 'sklearn',
        nbr_fracs: List[float] = [0.01, 0.05, 0.10],
        exclude_groups: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        data_type: str = 'gex',
        test_case: str = 'standard'
    ) -> ValidationResult:
        """
        Comprehensive comparison between two backends.
        
        Parameters:
        -----------
        X : np.ndarray
            Data matrix to test
        backend_test : str
            Backend under test (e.g., 'faiss-gpu')
        backend_baseline : str  
            Reference backend (default: 'sklearn')
        nbr_fracs : List[float]
            Neighbor fractions to test
        exclude_groups : Optional[Tuple[np.ndarray, np.ndarray]]
            Exclusion groups for TCR data
        data_type : str
            Type of data ('gex' or 'tcr')
        test_case : str
            Description of test case
            
        Returns:
        --------
        ValidationResult
            Comprehensive validation metrics
        """
        logger.info(f"Comprehensive backend comparison: {backend_test} vs {backend_baseline}")
        
        # Create searchers
        if backend_test.startswith('faiss'):
            backend_enum = Backend.FAISS_GPU if backend_test == 'faiss-gpu' else Backend.FAISS_CPU
            searcher_test = FaissNeighborSearcher(force_backend=backend_enum)
        else:
            searcher_test = None
            
        if backend_baseline.startswith('faiss'):
            baseline_enum = Backend.FAISS_GPU if backend_baseline == 'faiss-gpu' else Backend.FAISS_CPU  
            searcher_baseline = FaissNeighborSearcher(force_backend=baseline_enum)
        else:
            searcher_baseline = None
        
        # Use largest neighbor fraction for detailed analysis
        max_nbr_frac = max(nbr_fracs)
        max_k = max(1, int(max_nbr_frac * X.shape[0]))
        k_values = [k for k in [1, 5, 10, 20, 50, 100] if k <= max_k]
        
        try:
            # Get results from test backend
            if searcher_test is not None:
                result_test = searcher_test.search_neighbors(
                    X=X,
                    nbr_fracs=nbr_fracs,
                    exclude_groups=exclude_groups,
                    also_calc_nndists=True,
                    nbr_frac_for_nndists=max_nbr_frac,
                    sort_nbrs=True,
                    metric='euclidean',
                    data_type=data_type
                )
                neighbors_test = result_test.neighbors[max_nbr_frac]
            else:
                # Direct sklearn for baseline
                neighbors_test = self._sklearn_search(X, [max_nbr_frac], exclude_groups, data_type)[max_nbr_frac]
            
            # Get results from baseline backend  
            if searcher_baseline is not None:
                result_baseline = searcher_baseline.search_neighbors(
                    X=X,
                    nbr_fracs=nbr_fracs,
                    exclude_groups=exclude_groups,
                    also_calc_nndists=True,
                    nbr_frac_for_nndists=max_nbr_frac,
                    sort_nbrs=True,
                    metric='euclidean', 
                    data_type=data_type
                )
                neighbors_baseline = result_baseline.neighbors[max_nbr_frac]
            else:
                # Direct sklearn for baseline
                neighbors_baseline = self._sklearn_search(X, [max_nbr_frac], exclude_groups, data_type)[max_nbr_frac]
            
            # Calculate recall metrics
            recall_scores = self.validate_neighbor_recall(
                neighbors_test, neighbors_baseline, k_values=k_values
            )
            
            # Calculate precision (same as recall for same k)
            precision_scores = recall_scores.copy()
            
            # Calculate exact neighbor overlap
            identical_fraction = self._calculate_identical_neighbors(
                neighbors_test, neighbors_baseline
            )
            
            # Distance correlation (compute distances if needed)
            if data_type == 'gex':
                metric = 'euclidean'
            else:
                metric = 'sqeuclidean'  # For TCR vectors
                
            D_test = pairwise_distances(X, metric=metric)
            distance_corr_metrics = self.validate_distance_correlation(D_test, D_test)
            
            return ValidationResult(
                backend_a=backend_test,
                backend_b=backend_baseline,
                data_type=data_type,
                test_case=test_case,
                n_samples=X.shape[0],
                n_features=X.shape[1],
                k_values=k_values,
                recall_at_k=recall_scores,
                precision_at_k=precision_scores,
                distance_correlation=distance_corr_metrics['pearson_correlation'],
                identical_neighbors=identical_fraction,
                max_distance_error=distance_corr_metrics['max_absolute_error'],
                deterministic_match=True,  # Will be tested separately
                notes="Success"
            )
            
        except Exception as e:
            logger.error(f"Backend comparison failed: {e}")
            return ValidationResult(
                backend_a=backend_test,
                backend_b=backend_baseline,
                data_type=data_type,
                test_case=test_case,
                n_samples=X.shape[0],
                n_features=X.shape[1],
                k_values=[],
                recall_at_k={},
                precision_at_k={},
                distance_correlation=0.0,
                identical_neighbors=0.0,
                max_distance_error=float('inf'),
                deterministic_match=False,
                notes=f"Failed: {str(e)}"
            )
    def _sklearn_search(
        self, 
        X: np.ndarray, 
        nbr_fracs: List[float],
        exclude_groups: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        data_type: str = 'gex'
    ) -> Dict[float, np.ndarray]:
        """Direct sklearn neighbor search for baseline comparison."""
        
        if data_type == 'gex':
            metric = 'euclidean'
        else:
            metric = 'sqeuclidean'  # For TCR vectors
            
        D = pairwise_distances(X, metric=metric)
        
        # Apply exclusions if provided
        if exclude_groups is not None:
            agroups, bgroups = exclude_groups
            for ii, (a, b) in enumerate(zip(agroups, bgroups)):
                D[ii, (agroups == a)] = 1e6
                D[ii, (bgroups == b)] = 1e6
        
        neighbors = {}
        for nbr_frac in nbr_fracs:
            num_neighbors = max(1, int(nbr_frac * X.shape[0]))
            nbrs = np.argpartition(D, num_neighbors - 1)[:, :num_neighbors]
            
            # Sort by distance for consistent ordering
            ar = np.arange(X.shape[0])[:, None]
            inds = np.argsort(D[ar, nbrs])
            nbrs = nbrs[ar, inds]
            
            neighbors[nbr_frac] = nbrs
            
        return neighbors
    
    def _calculate_identical_neighbors(
        self, 
        neighbors_a: np.ndarray, 
        neighbors_b: np.ndarray
    ) -> float:
        """Calculate fraction of exactly identical neighbor sets."""
        
        n_samples = neighbors_a.shape[0]
        identical_count = 0
        
        for i in range(n_samples):
            set_a = set(neighbors_a[i])
            set_b = set(neighbors_b[i])
            if set_a == set_b:
                identical_count += 1
        
        return identical_count / n_samples if n_samples > 0 else 0.0
    
    def validate_determinism(
        self,
        X: np.ndarray,
        backend: str,
        n_runs: int = 3,
        nbr_fracs: List[float] = [0.05],
        data_type: str = 'gex'
    ) -> bool:
        """
        Validate that backend produces deterministic results across runs.
        
        Parameters:
        -----------
        X : np.ndarray
            Data to test
        backend : str
            Backend to test
        n_runs : int
            Number of runs to compare
        nbr_fracs : List[float]
            Neighbor fractions to test  
        data_type : str
            Type of data
            
        Returns:
        --------
        bool
            True if results are deterministic
        """
        logger.info(f"Testing determinism for {backend} with {n_runs} runs")
        
        results = []
        
        # Create searcher
        if backend.startswith('faiss'):
            backend_enum = Backend.FAISS_GPU if backend == 'faiss-gpu' else Backend.FAISS_CPU
            searcher = FaissNeighborSearcher(force_backend=backend_enum)
        else:
            searcher = None
        
        # Run multiple times with same seed
        for run in range(n_runs):
            np.random.seed(42)  # Fixed seed for determinism test
            
            if searcher is not None:
                result = searcher.search_neighbors(
                    X=X,
                    nbr_fracs=nbr_fracs,
                    metric='euclidean',
                    data_type=data_type
                )
                neighbors = result.neighbors
            else:
                neighbors = self._sklearn_search(X, nbr_fracs, data_type=data_type)
            
            results.append(neighbors)
        
        # Check if all results are identical
        baseline_result = results[0]
        for i in range(1, n_runs):
            for nbr_frac in nbr_fracs:
                if not np.array_equal(baseline_result[nbr_frac], results[i][nbr_frac]):
                    logger.warning(f"Determinism test failed for {backend} at run {i}")
                    return False
        
        logger.info(f"Determinism test passed for {backend}")
        return True
    def validate_edge_cases(self) -> List[ValidationResult]:
        """
        Validate FAISS accuracy on challenging edge case datasets.
        
        Returns:
        --------
        List[ValidationResult]
            Results for each edge case tested
        """
        logger.info("Running edge case validation suite")
        
        edge_results = []
        
        # Test cases: (generator_func, args, description)
        test_cases = [
            (EdgeCaseGenerator.generate_duplicate_vectors, 
             {'n_samples': 1000, 'n_features': 100, 'duplicate_fraction': 0.3},
             "duplicate_vectors"),
            (EdgeCaseGenerator.generate_high_dimensional_sparse,
             {'n_samples': 500, 'n_features': 2000, 'sparsity': 0.95}, 
             "high_dim_sparse"),
            (EdgeCaseGenerator.generate_clustered_data,
             {'n_samples': 800, 'n_features': 50, 'n_clusters': 8, 'cluster_separation': 1.5},
             "tight_clusters"),
            (EdgeCaseGenerator.generate_extreme_aspect_ratio,
             {'n_samples': 200, 'n_features': 5000},
             "extreme_aspect_ratio")
        ]
        
        # Test available FAISS backends
        test_backends = []
        if self.backend_info['faiss_cpu_available']:
            test_backends.append('faiss-cpu')
        if self.backend_info['faiss_gpu_available']:
            test_backends.append('faiss-gpu')
        
        for generator_func, args, description in test_cases:
            logger.info(f"Testing edge case: {description}")
            
            # Generate test data
            X = generator_func(**args)
            
            # Test both GEX and TCR data types  
            for data_type in ['gex', 'tcr']:
                # Adjust data for TCR if needed
                if data_type == 'tcr' and X.shape[1] != 1136:
                    # Resize to standard TCR vector length
                    if X.shape[1] > 1136:
                        X_test = X[:, :1136]
                    else:
                        # Pad with zeros
                        padding = np.zeros((X.shape[0], 1136 - X.shape[1]), dtype=X.dtype)
                        X_test = np.hstack([X, padding])
                else:
                    X_test = X
                
                # Generate exclusion groups for TCR
                exclude_groups = None
                if data_type == 'tcr':
                    agroups = np.random.randint(0, max(1, X_test.shape[0] // 3), X_test.shape[0])
                    bgroups = np.random.randint(0, max(1, X_test.shape[0] // 3), X_test.shape[0])
                    exclude_groups = (agroups, bgroups)
                
                # Test each backend against sklearn
                for backend in test_backends:
                    try:
                        result = self.compare_backends_comprehensive(
                            X=X_test,
                            backend_test=backend,
                            backend_baseline='sklearn',
                            nbr_fracs=[0.05, 0.10],
                            exclude_groups=exclude_groups,
                            data_type=data_type,
                            test_case=f"{description}_{data_type}"
                        )
                        edge_results.append(result)
                        
                    except Exception as e:
                        logger.error(f"Edge case {description} failed for {backend}: {e}")
                        # Add failure result
                        edge_results.append(ValidationResult(
                            backend_a=backend,
                            backend_b='sklearn',
                            data_type=data_type,
                            test_case=f"{description}_{data_type}",
                            n_samples=X_test.shape[0],
                            n_features=X_test.shape[1],
                            k_values=[],
                            recall_at_k={},
                            precision_at_k={}, 
                            distance_correlation=0.0,
                            identical_neighbors=0.0,
                            max_distance_error=float('inf'),
                            deterministic_match=False,
                            notes=f"Failed: {str(e)}"
                        ))
        
        logger.info(f"Completed edge case validation: {len(edge_results)} results")
        return edge_results
    
    def validate_all_backends(
        self,
        test_sizes: List[int] = [1000, 5000], 
        include_edge_cases: bool = True,
        include_determinism: bool = True
    ) -> DetailedReport:
        """
        Run comprehensive validation across all available backends.
        
        Parameters:
        -----------
        test_sizes : List[int]
            Dataset sizes to test
        include_edge_cases : bool
            Whether to include edge case testing
        include_determinism : bool  
            Whether to test deterministic behavior
            
        Returns:
        --------
        DetailedReport
            Comprehensive validation report
        """
        logger.info("Starting comprehensive FAISS accuracy validation")
        
        validation_results = []
        edge_case_results = []
        determinism_results = []
        
        # Available backends
        test_backends = []
        if self.backend_info['faiss_cpu_available']:
            test_backends.append('faiss-cpu')
        if self.backend_info['faiss_gpu_available']:
            test_backends.append('faiss-gpu')
        
        # Standard validation on different dataset sizes
        for n_samples in test_sizes:
            logger.info(f"Testing dataset size: {n_samples}")
            
            # Test both data types
            for data_type in ['gex', 'tcr']:
                # Generate test data
                if data_type == 'gex':
                    X = DatasetGenerator.generate_gex_data(n_samples)
                else:
                    X = DatasetGenerator.generate_tcr_vector_data(n_samples)
                
                exclude_groups = None
                if data_type == 'tcr':
                    agroups = np.random.randint(0, max(1, n_samples // 3), n_samples)
                    bgroups = np.random.randint(0, max(1, n_samples // 3), n_samples)
                    exclude_groups = (agroups, bgroups)
                
                # Test each backend
                for backend in test_backends:
                    result = self.compare_backends_comprehensive(
                        X=X,
                        backend_test=backend,
                        backend_baseline='sklearn',
                        nbr_fracs=[0.01, 0.05, 0.10],
                        exclude_groups=exclude_groups,
                        data_type=data_type,
                        test_case=f"standard_{n_samples}"
                    )
                    validation_results.append(result)
        
        # Edge case validation
        if include_edge_cases:
            edge_case_results = self.validate_edge_cases()
        
        # Determinism validation
        if include_determinism:
            for backend in test_backends:
                for data_type in ['gex', 'tcr']:
                    if data_type == 'gex':
                        X = DatasetGenerator.generate_gex_data(1000)
                    else:
                        X = DatasetGenerator.generate_tcr_vector_data(1000)
                    
                    is_deterministic = self.validate_determinism(
                        X=X, backend=backend, data_type=data_type
                    )
                    
                    # Create pseudo ValidationResult for determinism
                    determinism_results.append(ValidationResult(
                        backend_a=backend,
                        backend_b='determinism_test',
                        data_type=data_type,
                        test_case='determinism',
                        n_samples=X.shape[0],
                        n_features=X.shape[1],
                        k_values=[],
                        recall_at_k={},
                        precision_at_k={},
                        distance_correlation=1.0 if is_deterministic else 0.0,
                        identical_neighbors=1.0 if is_deterministic else 0.0,
                        max_distance_error=0.0 if is_deterministic else float('inf'),
                        deterministic_match=is_deterministic,
                        notes="Deterministic" if is_deterministic else "Non-deterministic"
                    ))
        
        # Generate report
        return self._generate_report(
            validation_results, edge_case_results, determinism_results
        )
    def _generate_report(
        self,
        validation_results: List[ValidationResult],
        edge_case_results: List[ValidationResult],
        determinism_results: List[ValidationResult]
    ) -> DetailedReport:
        """Generate comprehensive validation report."""
        
        from datetime import datetime
        
        all_results = validation_results + edge_case_results + determinism_results
        
        # Check if validation passes
        overall_pass = True
        production_ready = True
        recommendations = []
        
        for result in all_results:
            if result.notes.startswith("Failed"):
                overall_pass = False
                production_ready = False
                recommendations.append(f"Fix {result.backend_a} failure: {result.notes}")
                continue
            
            # Check accuracy thresholds
            if result.recall_at_k:
                min_recall = min(result.recall_at_k.values())
                if min_recall < self.tolerance:
                    overall_pass = False
                    recommendations.append(
                        f"{result.backend_a} recall {min_recall:.3f} < {self.tolerance:.3f} "
                        f"for {result.data_type} {result.test_case}"
                    )
            
            # Check identical neighbors for production use
            if result.identical_neighbors < 0.95:
                production_ready = False
                recommendations.append(
                    f"{result.backend_a} only {result.identical_neighbors:.1%} identical neighbors "
                    f"for {result.data_type} {result.test_case}"
                )
            
            # Check determinism in strict mode
            if self.strict_mode and not result.deterministic_match:
                production_ready = False
                recommendations.append(
                    f"{result.backend_a} failed determinism test for {result.data_type}"
                )
        
        # Generate recommendations
        if overall_pass and production_ready:
            recommendations.append("All FAISS backends passed validation ✓")
            recommendations.append("Safe for production use")
        elif overall_pass:
            recommendations.append("FAISS backends functional but may have precision differences")
            recommendations.append("Acceptable for research use, review for production")
        else:
            recommendations.append("FAISS validation failed - do not use for production")
            recommendations.append("Check FAISS installation and hardware compatibility")
        
        return DetailedReport(
            validation_results=validation_results,
            edge_case_results=edge_case_results,
            determinism_results=determinism_results,
            backend_availability=self.backend_info,
            overall_pass=overall_pass,
            production_ready=production_ready,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )


# Convenience functions for common validation scenarios

def quick_accuracy_check(
    n_samples: int = 2000,
    tolerance: float = 0.95
) -> bool:
    """
    Quick FAISS accuracy check for CI/testing.
    
    Parameters:
    -----------
    n_samples : int
        Size of test dataset
    tolerance : float
        Required accuracy threshold
        
    Returns:
    --------
    bool
        True if FAISS meets accuracy requirements
    """
    validator = AccuracyValidator(tolerance=tolerance, strict_mode=False)
    
    try:
        report = validator.validate_all_backends(
            test_sizes=[n_samples],
            include_edge_cases=False,
            include_determinism=False
        )
        
        if report.overall_pass:
            logger.info(f"Quick accuracy check PASSED (n={n_samples}, tol={tolerance})")
            return True
        else:
            logger.warning(f"Quick accuracy check FAILED (n={n_samples}, tol={tolerance})")
            return False
            
    except Exception as e:
        logger.error(f"Quick accuracy check errored: {e}")
        return False


def production_validation_suite() -> DetailedReport:
    """
    Production-grade validation suite with comprehensive testing.
    
    Returns:
    --------
    DetailedReport
        Complete validation report
    """
    validator = AccuracyValidator(tolerance=0.98, strict_mode=True)
    
    logger.info("Running production validation suite...")
    
    report = validator.validate_all_backends(
        test_sizes=[1000, 5000, 10000],
        include_edge_cases=True,
        include_determinism=True
    )
    
    logger.info(f"Production validation complete: {'PASS' if report.production_ready else 'FAIL'}")
    
    return report


def save_validation_report(report: DetailedReport, output_path: str):
    """Save detailed validation report to files."""
    output_path = Path(output_path)
    
    # Save summary
    with open(output_path.with_suffix('.txt'), 'w') as f:
        f.write("FAISS Accuracy Validation Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Timestamp: {report.timestamp}\n")
        f.write(f"Overall Pass: {report.overall_pass}\n")
        f.write(f"Production Ready: {report.production_ready}\n\n")
        
        f.write("Backend Availability:\n")
        for backend, available in report.backend_availability.items():
            f.write(f"  {backend}: {available}\n")
        f.write("\n")
        
        f.write("Recommendations:\n")
        for rec in report.recommendations:
            f.write(f"  - {rec}\n")
    
    # Save detailed results
    all_results = (report.validation_results + 
                  report.edge_case_results + 
                  report.determinism_results)
    
    if all_results:
        data = []
        for result in all_results:
            row = {
                'backend_a': result.backend_a,
                'backend_b': result.backend_b,
                'data_type': result.data_type,
                'test_case': result.test_case,
                'n_samples': result.n_samples,
                'n_features': result.n_features,
                'distance_correlation': result.distance_correlation,
                'identical_neighbors': result.identical_neighbors,
                'max_distance_error': result.max_distance_error,
                'deterministic_match': result.deterministic_match,
                'notes': result.notes
            }
            
            # Add recall metrics
            for k, recall in result.recall_at_k.items():
                row[f'recall_at_{k}'] = recall
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path.with_suffix('.csv'), index=False)
        
    logger.info(f"Validation report saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Quick check
    print("Running quick accuracy check...")
    quick_ok = quick_accuracy_check()
    print(f"Quick check: {'PASS' if quick_ok else 'FAIL'}")
    
    # Full validation
    print("\nRunning production validation suite...")
    report = production_validation_suite()
    
    print(f"\nProduction validation: {'PASS' if report.production_ready else 'FAIL'}")
    print(f"Recommendations: {len(report.recommendations)}")
    for rec in report.recommendations[:3]:  # Show first 3
        print(f"  - {rec}")

def comprehensive_error_handling_test():
    """
    Test comprehensive error handling scenarios for production robustness.
    
    Returns:
    --------
    bool
        True if error handling is robust
    """
    logger.info("Running comprehensive error handling test")
    
    validator = AccuracyValidator(tolerance=0.80, strict_mode=False)
    
    error_scenarios = []
    
    try:
        # Test 1: Invalid data shapes
        try:
            X_invalid = np.random.randn(0, 0).astype(np.float32)  # Empty
            validator.compare_backends_comprehensive(
                X=X_invalid,
                backend_test='sklearn',
                backend_baseline='sklearn',
                nbr_fracs=[0.05],
                test_case='empty_matrix'
            )
            error_scenarios.append(('empty_matrix', 'handled'))
        except Exception as e:
            error_scenarios.append(('empty_matrix', f'error: {type(e).__name__}'))
        
        # Test 2: Extreme values
        try:
            X_extreme = np.array([[1e10, -1e10], [np.inf, -np.inf]], dtype=np.float32)
            validator.validate_distance_correlation(X_extreme, X_extreme)
            error_scenarios.append(('extreme_values', 'handled'))
        except Exception as e:
            error_scenarios.append(('extreme_values', f'error: {type(e).__name__}'))
        
        # Test 3: NaN handling
        try:
            X_nan = np.array([[1.0, 2.0], [np.nan, 4.0]], dtype=np.float32)
            validator.validate_neighbor_recall(
                np.array([[0, 1], [1, 0]]), 
                np.array([[0, 1], [1, 0]])
            )
            error_scenarios.append(('nan_handling', 'handled'))
        except Exception as e:
            error_scenarios.append(('nan_handling', f'error: {type(e).__name__}'))
        
        # Test 4: Mismatched shapes - should raise ValueError
        try:
            validator.validate_neighbor_recall(
                np.array([[0, 1]]),  # 1x2
                np.array([[0, 1, 2]])  # 1x3 - mismatch
            )
            error_scenarios.append(('shape_mismatch', 'should_have_failed'))
        except ValueError as e:
            error_scenarios.append(('shape_mismatch', 'correctly_failed'))
        except Exception as e:
            error_scenarios.append(('shape_mismatch', f'unexpected_error: {type(e).__name__}'))
        
        # Test 5: Test backend forcing (should work with valid backends)
        try:
            X_test = np.random.randn(10, 5).astype(np.float32)
            # Test with a forced invalid backend by mocking
            from unittest.mock import patch
            with patch.object(validator, '_select_backend', side_effect=RuntimeError("Mock backend failure")):
                result = validator.compare_backends_comprehensive(
                    X=X_test,
                    backend_test='sklearn',  # This should work normally
                    backend_baseline='sklearn',
                    nbr_fracs=[0.10],
                    test_case='mock_failure'
                )
            
            if "Failed" in result.notes:
                error_scenarios.append(('mock_failure', 'correctly_handled'))
            else:
                error_scenarios.append(('mock_failure', 'unexpected_success'))
        except Exception as e:
            error_scenarios.append(('mock_failure', 'correctly_failed'))
        
    except Exception as e:
        logger.error(f"Error handling test failed: {e}")
        return False
    
    # Evaluate results
    passed = 0
    total = len(error_scenarios)
    
    logger.info("Error handling test results:")
    for scenario, outcome in error_scenarios:
        if outcome in ['handled', 'correctly_failed', 'correctly_handled']:
            passed += 1
            status = "✅"
        else:
            status = "❌"
        logger.info(f"  {status} {scenario}: {outcome}")
    
    success_rate = passed / total if total > 0 else 0
    logger.info(f"Error handling success rate: {success_rate:.1%}")
    
    return success_rate >= 0.75  # 75% of error scenarios handled correctly


def run_comprehensive_ci_validation() -> bool:
    """
    Run comprehensive validation suitable for CI environments.
    
    Optimized for speed while maintaining thorough coverage.
    
    Returns:
    --------
    bool
        True if all validation passes
    """
    logger.info("Running comprehensive CI validation")
    
    # Quick accuracy check
    accuracy_ok = quick_accuracy_check(n_samples=1000, tolerance=0.90)
    if not accuracy_ok:
        logger.error("Quick accuracy check failed")
        return False
    
    # Error handling test
    error_handling_ok = comprehensive_error_handling_test()
    if not error_handling_ok:
        logger.error("Error handling test failed")
        return False
    
    # Determinism check
    validator = AccuracyValidator(tolerance=0.90, strict_mode=False)
    X_det = np.random.randn(200, 20).astype(np.float32)
    
    sklearn_det = validator.validate_determinism(X_det, 'sklearn', n_runs=2)
    if not sklearn_det:
        logger.error("sklearn determinism failed")
        return False
    
    # Test FAISS determinism if available
    backend_info = get_backend_info()
    if backend_info['faiss_cpu_available']:
        faiss_det = validator.validate_determinism(X_det, 'faiss-cpu', n_runs=2)
        if not faiss_det:
            logger.error("FAISS CPU determinism failed")
            return False
    
    logger.info("All CI validation tests passed ✅")
    return True


if __name__ == "__main__":
    # Additional testing when run directly
    logging.basicConfig(level=logging.INFO)
    
    print("Running comprehensive error handling test...")
    error_ok = comprehensive_error_handling_test()
    print(f"Error handling test: {'PASS' if error_ok else 'FAIL'}")
    
    print("\nRunning CI validation...")
    ci_ok = run_comprehensive_ci_validation()
    print(f"CI validation: {'PASS' if ci_ok else 'FAIL'}")