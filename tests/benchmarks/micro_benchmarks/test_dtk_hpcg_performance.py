# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for DTK gpu-hpcg benchmark."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, Platform, ReturnCode
from superbench.benchmarks.result import BenchmarkResult


class DtkHpcgBenchmarkTest(BenchmarkTestCase, unittest.TestCase):
    """Tests for DTK gpu-hpcg benchmark."""

    example_raw_output = """
[1,0]<stdout>: rocHPCG version: 0.8.8-62f1830-dirty (based on hpcg-3.1)

[1,0]<stdout>: Setup Phase took 0.12 sec

[1,0]<stdout>: Starting Reference CG Phase ...


[1,0]<stdout>: Optimization Phase took 0.25 sec

[1,0]<stdout>: Validation Testing Phase ...

[1,0]<stdout>: Optimized CG Setup ...

[1,0]<stdout>: HIP Initial Residual = 2.668768e+04

[1,0]<stdout>: Total device memory usage: 19550 MByte (29152 MByte)

[1,0]<stdout>: Starting Benchmarking Phase ...

[1,0]<stdout>: Performing (at least) 2 CG sets in 1.0 seconds ...
[1,0]<stdout>: CG set 1 / 2    6881.2186 GFlop/s     (215.0381 GFlop/s per process)    50%    0.0 sec left
[1,0]<stdout>: CG set 2 / 2    6904.9453 GFlop/s     (215.7795 GFlop/s per process)    100%    0.0 sec left

[1,0]<stdout>: Local domain: 560 x 280 x 280
[1,0]<stdout>: Global domain: 2240 x 1120 x 560
[1,0]<stdout>: Process domain: 4 x 4 x 2

[1,0]<stdout>: Total Time: 7.55 sec
[1,0]<stdout>: Setup Time: 0.12 sec
[1,0]<stdout>: Optimization Time: 0.25 sec

[1,0]<stdout>: *** WARNING *** INVALID RUN

[1,0]<stdout>: DDOT   =  5849.4 GFlop/s ( 46794.9 GB/s)     182.8 GFlop/s per process ( 1462.3 GB/s per process)
[1,0]<stdout>: WAXPBY =  3052.0 GFlop/s ( 36623.8 GB/s)      95.4 GFlop/s per process ( 1144.5 GB/s per process)
[1,0]<stdout>: SpMV   =  5473.9 GFlop/s ( 34468.8 GB/s)     171.1 GFlop/s per process ( 1077.1 GB/s per process)
[1,0]<stdout>: MG     =  7716.9 GFlop/s ( 59557.1 GB/s)     241.2 GFlop/s per process ( 1861.2 GB/s per process)
[1,0]<stdout>: Total  =  6971.0 GFlop/s ( 52859.9 GB/s)     217.8 GFlop/s per process ( 1651.9 GB/s per process)
[1,0]<stdout>: Final  =  6904.9 GFlop/s ( 52359.0 GB/s)     215.8 GFlop/s per process ( 1636.2 GB/s per process)

[1,0]<stdout>: *** WARNING *** THIS IS NOT A VALID RUN ***
"""

    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.benchmark_name = 'gpu-hpcg'
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/run_rochpcg'])

    def get_benchmark(self):
        """Get benchmark."""
        (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, Platform.DTK)
        benchmark = benchmark_cls(self.benchmark_name, parameters='')
        benchmark._args = SimpleNamespace(
            log_raw_data=False,
            npx=4,
            npy=4,
            npz=2,
            nx=560,
            ny=280,
            nz=280,
        )
        benchmark._curr_run_index = 0
        benchmark._result = BenchmarkResult(self.benchmark_name, BenchmarkType.MICRO, ReturnCode.SUCCESS, run_count=1)
        return benchmark

    def test_dtk_hpcg_cls(self):
        """Test DTK gpu-hpcg benchmark class."""
        for platform in Platform:
            (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, platform)
            if platform is Platform.DTK:
                self.assertIsNotNone(benchmark_cls)
            else:
                self.assertIsNone(benchmark_cls)

    def test_dtk_hpcg_result_parsing_with_wrapper_noise(self):
        """Test DTK gpu-hpcg result parsing with wrapper noise."""
        benchmark = self.get_benchmark()

        self.assertTrue(benchmark._process_raw_result(0, self.example_raw_output))
        self.assertEqual(ReturnCode.SUCCESS, benchmark.return_code)

        workload = 'p4x4x2_n560x280x280'
        expected_results = {
            f'final_{workload}_flops': 6904.9,
            f'final_{workload}_flops_per_process': 215.8,
            f'final_{workload}_bandwidth': 52359.0,
            f'final_{workload}_bandwidth_per_process': 1636.2,
            f'ddot_{workload}_flops': 5849.4,
            f'ddot_{workload}_bandwidth': 46794.9,
            f'ddot_{workload}_flops_per_process': 182.8,
            f'ddot_{workload}_bandwidth_per_process': 1462.3,
            f'waxpby_{workload}_flops': 3052.0,
            f'waxpby_{workload}_bandwidth': 36623.8,
            f'waxpby_{workload}_flops_per_process': 95.4,
            f'waxpby_{workload}_bandwidth_per_process': 1144.5,
            f'spmv_{workload}_flops': 5473.9,
            f'spmv_{workload}_bandwidth': 34468.8,
            f'spmv_{workload}_flops_per_process': 171.1,
            f'spmv_{workload}_bandwidth_per_process': 1077.1,
            f'mg_{workload}_flops': 7716.9,
            f'mg_{workload}_bandwidth': 59557.1,
            f'mg_{workload}_flops_per_process': 241.2,
            f'mg_{workload}_bandwidth_per_process': 1861.2,
            f'total_{workload}_flops': 6971.0,
            f'total_{workload}_bandwidth': 52859.9,
            f'total_{workload}_flops_per_process': 217.8,
            f'total_{workload}_bandwidth_per_process': 1651.9,
            f'setup_time_{workload}': 0.12,
            f'optimization_time_{workload}': 0.25,
            f'total_time_{workload}': 7.55,
        }

        self.assertEqual(len(expected_results), len(benchmark.result) - benchmark.default_metric_count)
        for metric, value in expected_results.items():
            self.assertIn(metric, benchmark.result)
            self.assertEqual(value, benchmark.result[metric][0])
        for metric in benchmark.result:
            self.assertNotIn('valid', metric)
            self.assertNotIn('domain', metric)
        self.assertIn('raw_output_0', benchmark.raw_data)

    def test_dtk_hpcg_result_parsing_ignores_invalid_markers(self):
        """Test DTK gpu-hpcg does not emit validity metrics."""
        benchmark = self.get_benchmark()

        self.assertTrue(benchmark._process_raw_result(0, self.example_raw_output))
        self.assertFalse(any('valid' in metric for metric in benchmark.result))

    def test_dtk_hpcg_result_parsing_failure_when_required_summary_is_missing(self):
        """Test DTK gpu-hpcg parsing failure when required summary is missing."""
        benchmark = self.get_benchmark()
        invalid_output = self.example_raw_output.replace(
            '[1,0]<stdout>: Final  =  6904.9 GFlop/s ( 52359.0 GB/s)     '
            '215.8 GFlop/s per process ( 1636.2 GB/s per process)\n',
            '',
        )

        self.assertFalse(benchmark._process_raw_result(0, invalid_output))

    def test_dtk_hpcg_result_parsing_ignores_non_root_mpi_rank(self):
        """Test DTK gpu-hpcg parser skips non-root MPI ranks without summary output."""
        benchmark = self.get_benchmark()
        rank_only_output = '[1,2]<stdout>: [2]: Node Binding: Process 2 GPU: 2, NUMA: 0'

        with patch.dict(os.environ, {'OMPI_COMM_WORLD_RANK': '2'}):
            self.assertTrue(benchmark._process_raw_result(0, rank_only_output))
