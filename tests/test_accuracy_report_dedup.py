"""Unit test for Requirement 5.2 (prerequisite) of the tcrdist-db-update feature.

Confirms that the dead, duplicate first `accuracy_report` definition in
`conga/tcrdist/vectorized.py` -- which called the nonexistent
`TcrDistCalculator.tcr_distance` method and was unreachable because Python
keeps only the second module-scope definition of a repeated top-level name
-- was actually deleted, rather than merely shadowed by the second
definition still being present in the source.

Uses Python's `ast` module to parse the module source and count top-level
`FunctionDef` nodes named `accuracy_report`, which is robust to incidental
whitespace/formatting changes (unlike a plain `grep` of `^def accuracy_report`).
"""

import ast
import inspect

import conga.tcrdist.vectorized as vectorized


def _count_top_level_function_defs(module_source: str, name: str) -> int:
    """Count top-level (module-scope) FunctionDef/AsyncFunctionDef nodes named `name`."""
    tree = ast.parse(module_source)
    return sum(
        1
        for node in ast.iter_child_nodes(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )


def test_exactly_one_accuracy_report_definition_in_source():
    """The module source must contain exactly one top-level `accuracy_report` def."""
    source = inspect.getsource(vectorized)
    assert _count_top_level_function_defs(source, "accuracy_report") == 1


def test_accuracy_report_symbol_is_the_working_implementation():
    """The sole remaining `accuracy_report` must not reference the nonexistent
    `TcrDistCalculator.tcr_distance` method that made the dead first
    definition unreachable dead code in the first place.
    """
    source = inspect.getsource(vectorized.accuracy_report)
    assert "calculator.tcr_distance(" not in source
