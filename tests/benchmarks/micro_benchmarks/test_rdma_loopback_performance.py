# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for rdma-loopback benchmark."""

import os
import numbers
import unittest
from unittest import mock

from tests.helper import decorator
from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, Platform, BenchmarkType, ReturnCode
from superbench.common.utils import network
from superbench.benchmarks.micro_benchmarks import rdma_loopback_performance


class RdmaLoopbackBenchmarkTest(BenchmarkTestCase, unittest.TestCase):
    """Tests for RdmaLoopbackBenchmark benchmark."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/run_perftest_loopback'])

    def test_rdma_loopback_util(self):
        """Test util functions 'get_numa_cores' and 'get_free_port' used in rdma-loopback benchmark."""
        port = network.get_free_port()
        assert (isinstance(port, numbers.Number))
        numa_cores = rdma_loopback_performance.get_numa_cores(0)
        if numa_cores is None:
            # in case no NUMA support available on test system
            return
        assert (len(numa_cores) >= 2)
        for i in range(len(numa_cores)):
            assert (isinstance(numa_cores[i], numbers.Number))

    @decorator.load_data('tests/data/ib_loopback_all_sizes.log')
    @mock.patch('superbench.benchmarks.micro_benchmarks.rdma_loopback_performance.get_numa_cores')
    @mock.patch('superbench.common.utils.network.get_rdma_devices')
    def test_rdma_loopback_all_sizes(self, raw_output, mock_rdma_devices, mock_numa_cores):
        """Test rdma-loopback benchmark for all sizes."""
        benchmark_name = 'rdma-loopback'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)

        parameters = '--iters 2000'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        ret = benchmark._preprocess()
        assert (ret is False)
        assert (benchmark.return_code == ReturnCode.INVALID_ARGUMENT)

        parameters = '--rdma_mapping 0::1 --link_layer ethernet --iters 2000'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        mock_rdma_devices.return_value = []
        ret = benchmark._preprocess()
        assert (ret is False)
        assert (benchmark.return_code == ReturnCode.MICROBENCHMARK_DEVICE_GETTING_FAILURE)

        parameters = '--rdma_mapping mlx5_0::1 --link_layer ethernet --iters 2000'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        mock_rdma_devices.return_value = [
            {
                'device': 'mlx5_0',
                'port': '1',
                'link_layer': 'InfiniBand',
                'state': '4: ACTIVE',
                'phys_state': '5: LinkUp',
                'netdevs': [],
            }
        ]
        ret = benchmark._preprocess()
        assert (ret is False)
        assert (benchmark.return_code == ReturnCode.MICROBENCHMARK_DEVICE_GETTING_FAILURE)

        parameters = '--rdma_mapping mlx5_0::1 --iters 2000'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        mock_numa_cores.return_value = [0, 1, 2, 3]
        ret = benchmark._preprocess()
        assert (ret)

        port = benchmark._RdmaLoopbackBenchmark__sock_fds[-1].getsockname()[1]
        expect_command = 'run_perftest_loopback 3 1 ' + benchmark._args.bin_dir + \
            f'/ib_write_bw -a -F --iters=2000 -d mlx5_0 -p {port} -x 0 --report_gbits'
        command = benchmark._bin_name + benchmark._commands[0].split(benchmark._bin_name)[1]
        assert (command == expect_command)

        assert (benchmark._process_raw_result(0, raw_output))

        metric_list = []
        for rdma_command in benchmark._args.commands:
            for size in ['8388608', '4194304', '1024', '2']:
                metric = 'ib_{}_bw_{}:mlx5_0'.format(rdma_command, size)
                metric_list.append(metric)
        for metric in metric_list:
            assert (metric in benchmark.result)
            assert (len(benchmark.result[metric]) == 1)
            assert (isinstance(benchmark.result[metric][0], numbers.Number))

        assert (benchmark._process_raw_result(0, 'Invalid raw output') is False)

        assert (benchmark.name == 'rdma-loopback')
        assert (benchmark.type == BenchmarkType.MICRO)
        assert (benchmark._bin_name == 'run_perftest_loopback')

        assert (benchmark._args.iters == 2000)
        assert (benchmark._args.commands == ['write'])

        parameters = '--rdma_mapping 0::1 --link_layer ethernet --iters 2000'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        mock_rdma_devices.return_value = [
            {
                'device': 'mlx5_7',
                'port': '1',
                'link_layer': 'Ethernet',
                'state': '4: ACTIVE',
                'phys_state': '5: LinkUp',
                'netdevs': ['eth0'],
            }
        ]
        mock_numa_cores.return_value = [0, 1, 2, 3]
        ret = benchmark._preprocess()
        assert (ret)
        port = benchmark._RdmaLoopbackBenchmark__sock_fds[-1].getsockname()[1]
        expect_command = 'run_perftest_loopback 3 1 ' + benchmark._args.bin_dir + \
            f'/ib_write_bw -a -F --iters=2000 -d mlx5_7 -p {port} -x 0 --report_gbits'
        command = benchmark._bin_name + benchmark._commands[0].split(benchmark._bin_name)[1]
        assert (command == expect_command)
        mock_rdma_devices.assert_any_call()

        parameters = '--rdma_mapping 1::1 --iters 2000'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        mock_rdma_devices.return_value = [
            {
                'device': 'mlx5_0',
                'port': '1',
                'link_layer': 'Ethernet',
                'state': '4: ACTIVE',
                'phys_state': '5: LinkUp',
                'netdevs': ['eth0'],
            },
            {
                'device': 'mlx5_0',
                'port': '2',
                'link_layer': 'Ethernet',
                'state': '4: ACTIVE',
                'phys_state': '5: LinkUp',
                'netdevs': ['eth1'],
            },
            {
                'device': 'mlx5_1',
                'port': '1',
                'link_layer': 'Ethernet',
                'state': '4: ACTIVE',
                'phys_state': '5: LinkUp',
                'netdevs': ['eth2'],
            },
        ]
        mock_numa_cores.return_value = [0, 1, 2, 3]
        ret = benchmark._preprocess()
        assert (ret)
        port = benchmark._RdmaLoopbackBenchmark__sock_fds[-1].getsockname()[1]
        expect_command = 'run_perftest_loopback 3 1 ' + benchmark._args.bin_dir + \
            f'/ib_write_bw -a -F --iters=2000 -d mlx5_1 -p {port} -x 0 --report_gbits'
        command = benchmark._bin_name + benchmark._commands[0].split(benchmark._bin_name)[1]
        assert (command == expect_command)

    @decorator.load_data('tests/data/ib_loopback_8M_size.log')
    @mock.patch('superbench.benchmarks.micro_benchmarks.rdma_loopback_performance.get_numa_cores')
    @mock.patch('superbench.common.utils.network.get_rdma_devices')
    def test_rdma_loopback_8M_size(self, raw_output, mock_rdma_devices, mock_numa_cores):
        """Test rdma-loopback benchmark for 8M size."""
        benchmark_name = 'rdma-loopback'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)

        parameters = (
            '--rdma_mapping mlx5_2:0:0 --duration 15 --qp 2 --bidirectional '
            '--tclass 96 --msg_size 8388608 --gpu_backend rocm --gpu_dmabuf'
        )
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        mock_numa_cores.return_value = [0, 1, 2, 3]
        ret = benchmark._preprocess()
        assert (ret)

        port = benchmark._RdmaLoopbackBenchmark__sock_fds[-1].getsockname()[1]
        expect_command = 'run_perftest_loopback 3 1 ' + benchmark._args.bin_dir + \
            f'/ib_write_bw -s 8388608 -F -D 15 -d mlx5_2 -p {port} -x 0 --report_gbits ' + \
            '-q 2 --tclass 96 -b --use_rocm 0 --use_rocm_dmabuf'
        command = benchmark._bin_name + benchmark._commands[0].split(benchmark._bin_name)[1]
        assert (command == expect_command)

        assert (benchmark._process_raw_result(0, raw_output))

        metric = 'ib_write_bw_8388608:mlx5_2'
        assert (metric in benchmark.result)
        assert (len(benchmark.result[metric]) == 1)
        assert (isinstance(benchmark.result[metric][0], numbers.Number))

        assert (benchmark._process_raw_result(0, 'Invalid raw output') is False)

        assert (benchmark.name == 'rdma-loopback')
        assert (benchmark.type == BenchmarkType.MICRO)
        assert (benchmark._bin_name == 'run_perftest_loopback')

        assert (benchmark._args.duration == 15)
        assert (benchmark._args.qp == 2)
        assert (benchmark._args.bidirectional)
        assert (benchmark._args.tclass == 96)
        assert (benchmark._args.msg_size == 8388608)
        assert (benchmark._args.gpu_backend == 'rocm')
        assert (benchmark._args.gpu_dmabuf)
        assert (benchmark._args.commands == ['write'])

    @mock.patch('superbench.benchmarks.micro_benchmarks.rdma_loopback_performance.get_numa_cores')
    def test_rdma_loopback_rank_mapping(self, mock_numa_cores):
        """Test rdma-loopback benchmark with explicit RDMA/GPU/NUMA rank mapping."""
        benchmark_name = 'rdma-loopback'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)

        parameters = (
            '--rdma_mapping mlx5_2:0:3 mlx5_4:3:0 '
            '--gpu_backend cuda --msg_size 8388608 --cuda_mem_type 0 --gpu_touch once'
        )
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        mock_numa_cores.return_value = [0, 1, 2, 3]
        os.environ['PROC_RANK'] = '1'
        ret = benchmark._preprocess()
        assert (ret)

        port = benchmark._RdmaLoopbackBenchmark__sock_fds[-1].getsockname()[1]
        expect_command = 'run_perftest_loopback 3 1 ' + benchmark._args.bin_dir + \
            f'/ib_write_bw -s 8388608 -F --iters=20000 -d mlx5_4 -p {port} -x 0 --report_gbits ' + \
            '--use_cuda 3 --cuda_mem_type 0 --gpu_touch once'
        command = benchmark._bin_name + benchmark._commands[0].split(benchmark._bin_name)[1]
        assert (command == expect_command)

        os.environ.pop('PROC_RANK', None)

    @mock.patch(
        'superbench.benchmarks.micro_benchmarks.rdma_loopback_performance.RdmaLoopbackBenchmark._get_affinity_cores'
    )
    def test_rdma_loopback_auto_cpu_cores(self, mock_affinity_cores):
        """Test rdma-loopback benchmark with automatic CPU core selection."""
        benchmark_name = 'rdma-loopback'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)

        parameters = '--rdma_mapping mlx5_2 --msg_size 8388608'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        mock_affinity_cores.return_value = [4, 5, 6, 7]
        ret = benchmark._preprocess()
        assert (ret)

        port = benchmark._RdmaLoopbackBenchmark__sock_fds[-1].getsockname()[1]
        expect_command = 'run_perftest_loopback 7 5 ' + benchmark._args.bin_dir + \
            f'/ib_write_bw -s 8388608 -F --iters=20000 -d mlx5_2 -p {port} -x 0 --report_gbits'
        command = benchmark._bin_name + benchmark._commands[0].split(benchmark._bin_name)[1]
        assert (command == expect_command)

    def test_rdma_loopback_invalid_rank_mapping(self):
        """Test invalid RDMA rank mapping arguments."""
        benchmark_name = 'rdma-loopback'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)

        benchmark = benchmark_class(benchmark_name, parameters='--rdma_mapping mlx5_2:')
        ret = benchmark._preprocess()
        assert (ret is False)
        assert (benchmark.return_code == ReturnCode.INVALID_ARGUMENT)

        benchmark = benchmark_class(benchmark_name, parameters='--rdma_mapping mlx5_2::')
        ret = benchmark._preprocess()
        assert (ret is False)
        assert (benchmark.return_code == ReturnCode.INVALID_ARGUMENT)

        benchmark = benchmark_class(benchmark_name, parameters='--rdma_mapping mlx5_2::3 --gpu_backend rocm')
        ret = benchmark._preprocess()
        assert (ret is False)
        assert (benchmark.return_code == ReturnCode.INVALID_ARGUMENT)

        benchmark = benchmark_class(benchmark_name, parameters='--rdma_mapping mlx5_2:0:3 --gpu_backend none')
        ret = benchmark._preprocess()
        assert (ret is False)
        assert (benchmark.return_code == ReturnCode.INVALID_ARGUMENT)

        benchmark = benchmark_class(benchmark_name, parameters='--rdma_mapping mlx5_2::3 --numa 3')
        ret = benchmark._preprocess()
        assert (ret is False)
        assert (benchmark.return_code == ReturnCode.INVALID_ARGUMENT)
