# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Metric sort helpers for analyzer outputs.

This module keeps benchmark-specific metric ordering isolated from the generic
summary generation flow. Benchmarks without a registered sorter fall back to
plain string ordering.
"""

import re

_RCCL_PATTERN = re.compile(r'^(?P<bench>rccl-bw(?::[^/]+)?)/(?P<op>[^_]+)_(?P<size>\d+)_(?P<suffix>.+?)(?::\d+)?$')
_HPCG_PATTERN = re.compile(r'^(?P<bench>gpu-hpcg(?::[^/]+)?)/(?P<metric>.+?)(?::\d+)?$')
_HPCG_WORKLOAD_PATTERN = re.compile(
    r'^(?P<subject>final|ddot|waxpby|spmv|mg|total)_'
    r'p(?P<npx>\d+)x(?P<npy>\d+)x(?P<npz>\d+)_'
    r'n(?P<nx>\d+)x(?P<ny>\d+)x(?P<nz>\d+)_'
    r'(?P<type>gflops|bandwidth|gflops_per_process|bandwidth_per_process)$'
)
_HPCG_TIME_PATTERN = re.compile(
    r'^(?P<subject>setup_time|optimization_time|total_time)_'
    r'p(?P<npx>\d+)x(?P<npy>\d+)x(?P<npz>\d+)_'
    r'n(?P<nx>\d+)x(?P<ny>\d+)x(?P<nz>\d+)$'
)

_HPCG_SUBJECT_ORDER = {
    'setup_time': 0,
    'optimization_time': 1,
    'total_time': 2,
    'ddot': 3,
    'waxpby': 4,
    'spmv': 5,
    'mg': 6,
    'total': 7,
    'final': 8,
}

_HPCG_PERF_TYPE_ORDER = {
    'gflops': 0,
    'bandwidth': 1,
    'gflops_per_process': 2,
    'bandwidth_per_process': 3,
}


def _rccl_sort_key(metric_name):
    """Sort RCCL metrics by benchmark, operation, then numeric message size."""
    match = _RCCL_PATTERN.match(metric_name)
    if not match:
        return None

    return (
        0,
        match.group('bench'),
        match.group('op'),
        int(match.group('size')),
        match.group('suffix'),
        metric_name,
    )


def _hpcg_workload_key(match):
    """Return a numeric sort key for the HPCG process domain and local problem size."""
    return (
        int(match.group('npx')),
        int(match.group('npy')),
        int(match.group('npz')),
        int(match.group('nx')),
        int(match.group('ny')),
        int(match.group('nz')),
    )


def _hpcg_sort_key(metric_name):
    """Sort HPCG metrics roughly in the order they appear in rocHPCG logs."""
    match = _HPCG_PATTERN.match(metric_name)
    if not match:
        return None

    metric = match.group('metric')
    time_match = _HPCG_TIME_PATTERN.match(metric)
    if time_match:
        return (
            1,
            match.group('bench'),
            _HPCG_SUBJECT_ORDER.get(time_match.group('subject'), 999),
            0,
            *_hpcg_workload_key(time_match),
            metric_name,
        )

    workload_match = _HPCG_WORKLOAD_PATTERN.match(metric)
    if workload_match:
        subject = workload_match.group('subject')
        metric_type = workload_match.group('type')
        return (
            1,
            match.group('bench'),
            _HPCG_SUBJECT_ORDER.get(subject, 999),
            _HPCG_PERF_TYPE_ORDER.get(metric_type, 999),
            *_hpcg_workload_key(workload_match),
            metric_name,
        )

    return (
        1,
        match.group('bench'),
        _HPCG_SUBJECT_ORDER.get(metric, 999),
        metric,
        metric_name,
    )


_SORTERS = (
    _rccl_sort_key,
    _hpcg_sort_key,
)


def sort_metrics(metrics):
    """Sort metrics with benchmark-specific sorters and a stable default fallback."""
    def sort_key(metric_name):
        for sorter in _SORTERS:
            key = sorter(metric_name)
            if key is not None:
                return key
        return (999, metric_name)

    return sorted(metrics, key=sort_key)
