# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the RDMA loopback benchmarks."""

import os
import socket
from pathlib import Path

from superbench.common.utils import logger
from superbench.common.utils import network
from superbench.benchmarks import BenchmarkRegistry, ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


def get_numa_cores(numa_index):
    """Get the available cores from different physical cpu core of NUMA<numa_index>.

    Args:
        numa_index (int): the index of numa node.

    Return:
        list: The available cores from different physical cpu core of NUMA<numa_index>.
        None if no available cores or numa index.
    """
    try:
        with Path(f'/sys/devices/system/node/node{numa_index}/cpulist').open('r') as f:
            cores = []
            core_ranges = f.read().strip().split(',')
            for core_range in core_ranges:
                start, end = core_range.split('-')
                for core in range(int(start), int(end) + 1):
                    cores.append(core)
        return cores
    except IOError:
        return None


class RdmaLoopbackBenchmark(MicroBenchmarkWithInvoke):
    """The RDMA loopback performance benchmark class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = 'run_perftest_loopback'
        self.__sock_fds = []
        self.__support_rdma_commands = {'write': 'ib_write_bw', 'read': 'ib_read_bw', 'send': 'ib_send_bw'}
        self.__support_gpu_backends = ['none', 'cuda', 'rocm']
        self.__support_link_layers = ['infiniband', 'ethernet']
        self.__rdma_device = None
        self.__gpu_device = None
        self.__numa_node = None

    def __del__(self):
        """Destructor."""
        for fd in self.__sock_fds:
            fd.close()

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--link_layer',
            type=str,
            default=None,
            required=False,
            choices=self.__support_link_layers,
            help='The expected RDMA link layer.',
        )
        self._parser.add_argument(
            '--rdma_mapping',
            type=str,
            nargs='+',
            default=None,
            required=False,
            help='The rank mapping entries in RDMA[:GPU[:NUMA]] format, e.g., mlx5_2:0:3 mlx5_4:1.',
        )
        self._parser.add_argument(
            '--iters',
            type=int,
            default=20000,
            required=False,
            help='The iterations of running RDMA command',
        )
        self._parser.add_argument(
            '--qp',
            type=int,
            default=None,
            required=False,
            help='The number of queue pairs.',
        )
        self._parser.add_argument(
            '--bidirectional',
            action='store_true',
            default=False,
            help='Measure bidirectional bandwidth.',
        )
        self._parser.add_argument(
            '--tclass',
            type=int,
            default=None,
            required=False,
            help='The traffic class used by perftest.',
        )
        self._parser.add_argument(
            '--msg_size',
            type=int,
            default=None,
            required=False,
            help='The message size of running RDMA command, e.g., 8388608.',
        )
        self._parser.add_argument(
            '--commands',
            type=str,
            nargs='+',
            default=['write'],
            help='The RDMA command used to run, e.g., {}.'.format(' '.join(list(self.__support_rdma_commands.keys()))),
        )
        self._parser.add_argument(
            '--gid_index',
            type=int,
            default=0,
            required=False,
            help='Test uses GID with GID index taken from command.',
        )
        self._parser.add_argument(
            '--extra_perftest_args',
            type=str,
            default=None,
            required=False,
            help='Extra arguments appended to the perftest command.',
        )
        self._parser.add_argument(
            '--gpu_backend',
            type=str,
            default='none',
            required=False,
            choices=self.__support_gpu_backends,
            help='The GPU backend used for GPUDirect RDMA testing.',
        )
        self._parser.add_argument(
            '--gpu_dmabuf',
            action='store_true',
            default=False,
            help='Use GPU DMA-BUF for GPUDirect RDMA testing.',
        )
        self._parser.add_argument(
            '--cuda_mem_type',
            type=int,
            default=None,
            required=False,
            help='Set CUDA memory type for perftest.',
        )
        self._parser.add_argument(
            '--gpu_touch',
            type=str,
            default=None,
            required=False,
            choices=['once', 'infinite'],
            help='Touch GPU pages before GPUDirect RDMA testing.',
        )

    def _get_proc_rank(self):
        """Get the rank of current process."""
        if os.getenv('PROC_RANK'):
            return int(os.getenv('PROC_RANK'))
        return 0

    def _get_rdma_mapping_entries(self):
        """Normalize RDMA mapping argument to a list of entries."""
        entries = []
        if not self._args.rdma_mapping:
            return entries
        for mapping_arg in self._args.rdma_mapping:
            entries += [entry.strip() for entry in mapping_arg.split(',') if entry.strip()]
        return entries

    def _resolve_rdma_ref(self, rdma_ref):
        """Resolve RDMA mapping reference to device name."""
        if not rdma_ref.isdigit() and not self._args.link_layer:
            return rdma_ref

        try:
            rdma_devices = network.get_rdma_devices()
        except BaseException as e:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_DEVICE_GETTING_FAILURE)
            logger.error('Getting RDMA devices failure - benchmark: {}, message: {}.'.format(self._name, str(e)))
            return None

        rdma_device = None
        if rdma_ref.isdigit():
            rdma_index = int(rdma_ref)
            if rdma_devices and 0 <= rdma_index < len(rdma_devices):
                rdma_device = rdma_devices[rdma_index]
        elif rdma_devices:
            rdma_device = next((device for device in rdma_devices if device['device'] == rdma_ref), None)

        if not rdma_device:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_DEVICE_GETTING_FAILURE)
            logger.error(
                'Getting RDMA devices failure - benchmark: {}, device: {}, devices: {}.'.format(
                    self._name, rdma_ref, rdma_devices
                )
            )
            return None

        if self._args.link_layer and rdma_device['link_layer'].lower() != self._args.link_layer:
            self._result.set_return_code(ReturnCode.MICROBENCHMARK_DEVICE_GETTING_FAILURE)
            logger.error(
                'RDMA link layer mismatch - benchmark: {}, device: {}, expected: {}, actual: {}.'.format(
                    self._name, rdma_device['device'], self._args.link_layer, rdma_device['link_layer']
                )
            )
            return None
        return rdma_device['device']

    def _parse_rdma_mapping_entry(self, entry):
        """Parse one RDMA mapping entry and apply it to the benchmark."""
        fields = entry.split(':')
        if len(fields) > 3 or not fields[0]:
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            logger.error(
                'Invalid RDMA mapping entry - benchmark: {}, entry: {}, expected format: RDMA[:GPU[:NUMA]].'.format(
                    self._name, entry
                )
            )
            return False
        if len(fields) == 2 and not fields[1]:
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            logger.error('Invalid RDMA mapping entry - benchmark: {}, entry: {}.'.format(self._name, entry))
            return False
        if len(fields) == 3 and not fields[2]:
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            logger.error('Invalid RDMA mapping entry - benchmark: {}, entry: {}.'.format(self._name, entry))
            return False

        self.__rdma_device = self._resolve_rdma_ref(fields[0])
        if not self.__rdma_device:
            return False

        try:
            self.__gpu_device = int(fields[1]) if len(fields) >= 2 and fields[1] else None
            self.__numa_node = int(fields[2]) if len(fields) == 3 else None
        except BaseException:
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            logger.error(
                'Invalid RDMA mapping value - benchmark: {}, entry: {}, expected numeric GPU and NUMA.'.format(
                    self._name, entry
                )
            )
            return False
        if self.__gpu_device is not None and self.__gpu_device < 0:
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            logger.error('GPU device id must be non-negative - benchmark: {}, entry: {}.'.format(self._name, entry))
            return False
        if self.__numa_node is not None and self.__numa_node < 0:
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            logger.error('NUMA node must be non-negative - benchmark: {}, entry: {}.'.format(self._name, entry))
            return False
        return True

    def _apply_rdma_mapping(self):
        """Apply the current rank's RDMA/GPU/NUMA mapping."""
        entries = self._get_rdma_mapping_entries()
        if not entries:
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            logger.error('RDMA mapping is required - benchmark: {}.'.format(self._name))
            return False

        try:
            rank = self._get_proc_rank()
        except BaseException:
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            logger.error('Invalid PROC_RANK - benchmark: {}, PROC_RANK: {}.'.format(self._name, os.getenv('PROC_RANK')))
            return False

        if rank < 0 or rank >= len(entries):
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            logger.error(
                'The rank is out of RDMA mapping entries - benchmark: {}, rank: {}, rdma_mapping: {}.'.format(
                    self._name, rank, entries
                )
            )
            return False

        return self._parse_rdma_mapping_entry(entries[rank])

    def _bind_free_port(self):
        """Bind a local TCP socket to a free port and keep it reserved."""
        try:
            self.__sock_fds.append(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
            # grep SO_REUSE /usr/include/asm-generic/socket.h
            self.__sock_fds[-1].setsockopt(socket.SOL_SOCKET, getattr(socket, 'SO_REUSEADDR', 2), 1)
            self.__sock_fds[-1].setsockopt(socket.SOL_SOCKET, getattr(socket, 'SO_REUSEPORT', 15), 1)
            self.__sock_fds[-1].bind(('127.0.0.1', 0))
        except OSError as e:
            self._result.set_return_code(ReturnCode.RUNTIME_EXCEPTION_ERROR)
            logger.error('Error when binding port - benchmark: %s, message: %s.', self._name, e)
            return False
        return True

    def _get_affinity_cores(self):
        """Get available CPU cores from current process affinity."""
        try:
            return sorted(os.sched_getaffinity(0))
        except AttributeError:
            cpu_count = os.cpu_count()
            return list(range(cpu_count)) if cpu_count else None

    def _get_cpu_cores(self):
        """Get CPU cores from NUMA node or current process affinity."""
        if self.__numa_node is not None:
            return get_numa_cores(self.__numa_node)
        return self._get_affinity_cores()

    def _get_command_mode(self):
        """Get perftest message size mode."""
        if self._args.msg_size is None:
            return ' -a'
        return ' -s ' + str(self._args.msg_size)

    def _get_run_mode(self):
        """Get perftest run-length mode."""
        if self._args.duration > 0:
            return ' -D ' + str(self._args.duration)
        return ' --iters=' + str(self._args.iters)

    def _get_metric_suffix(self):
        """Get suffix used for RDMA loopback metrics."""
        return self.__rdma_device

    def _build_perftest_options(self, rdma_device):
        """Build perftest command options."""
        command = self._get_command_mode() + ' -F'
        command += self._get_run_mode()
        command += ' -d ' + rdma_device
        command += ' -p ' + str(self.__sock_fds[-1].getsockname()[1])
        command += ' -x ' + str(self._args.gid_index)
        command += ' --report_gbits'
        if self._args.qp is not None:
            command += ' -q ' + str(self._args.qp)
        if self._args.tclass is not None:
            command += ' --tclass ' + str(self._args.tclass)
        if self._args.bidirectional:
            command += ' -b'
        if self._args.extra_perftest_args:
            command += ' ' + self._args.extra_perftest_args
        return command

    def _validate_gpu_arguments(self):
        """Validate GPU memory arguments."""
        if self._args.gpu_backend == 'none':
            if self.__gpu_device is not None:
                self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
                logger.error('GPU device requires GPU backend - benchmark: {}.'.format(self._name))
                return False
            if self._args.gpu_dmabuf or self._args.cuda_mem_type is not None or self._args.gpu_touch:
                self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
                logger.error('GPU options require GPU backend - benchmark: {}.'.format(self._name))
                return False
        else:
            if self.__gpu_device is None:
                self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
                logger.error('GPU backend requires GPU device - benchmark: {}.'.format(self._name))
                return False
            if self._args.cuda_mem_type is not None and self._args.gpu_backend != 'cuda':
                self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
                logger.error('CUDA memory type requires CUDA backend - benchmark: {}.'.format(self._name))
                return False
        return True

    def _build_gpu_options(self):
        """Build perftest GPU memory options."""
        if self._args.gpu_backend == 'none':
            return ''

        command = ' --use_' + self._args.gpu_backend + ' ' + str(self.__gpu_device)
        if self._args.gpu_dmabuf:
            command += ' --use_' + self._args.gpu_backend + '_dmabuf'
        if self._args.cuda_mem_type is not None:
            command += ' --cuda_mem_type ' + str(self._args.cuda_mem_type)
        if self._args.gpu_touch:
            command += ' --gpu_touch ' + self._args.gpu_touch
        return command

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False
        if self._args.numa is not None:
            self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
            logger.error('Use RDMA mapping to set NUMA node - benchmark: {}.'.format(self._name))
            return False
        if not self._apply_rdma_mapping():
            return False
        if not self._validate_gpu_arguments():
            return False

        self._args.commands = [command.lower() for command in self._args.commands]

        for rdma_command in self._args.commands:
            if rdma_command not in self.__support_rdma_commands:
                self._result.set_return_code(ReturnCode.INVALID_ARGUMENT)
                logger.error(
                    'Unsupported RDMA command - benchmark: {}, command: {}, expected: {}.'.format(
                        self._name, rdma_command, ' '.join(list(self.__support_rdma_commands.keys()))
                    )
                )
                return False

            if not self._bind_free_port():
                return False
            cpu_cores = self._get_cpu_cores()
            if not cpu_cores or len(cpu_cores) < 2:
                self._result.set_return_code(ReturnCode.MICROBENCHMARK_DEVICE_GETTING_FAILURE)
                logger.error('Getting CPU core devices failure - benchmark: {}.'.format(self._name))
                return False
            command = os.path.join(self._args.bin_dir, self._bin_name)
            command += ' ' + str(cpu_cores[-1]) + ' ' + str(cpu_cores[-3 + int((len(cpu_cores) < 4))])
            command += ' ' + os.path.join(self._args.bin_dir, self.__support_rdma_commands[rdma_command])
            command += self._build_perftest_options(self.__rdma_device)
            command += self._build_gpu_options()
            self._commands.append(command)

        return True

    def _process_raw_result(self, cmd_idx, raw_output):
        """Function to parse raw results and save the summarized results.

          self._result.add_raw_data() and self._result.add_result() need to be called to save the results.

        Args:
            cmd_idx (int): the index of command corresponding with the raw_output.
            raw_output (str): raw output string of the micro-benchmark.

        Return:
            True if the raw output string is valid and result can be extracted.
        """
        self._result.add_raw_data(
            'raw_output_' + self._args.commands[cmd_idx] + '_RDMA' + self._get_metric_suffix(), raw_output,
            self._args.log_raw_data
        )

        valid = False
        content = raw_output.splitlines()

        metric_set = set()
        for line in content:
            try:
                values = list(filter(None, line.split()))
                if len(values) != 5:
                    continue
                size = int(values[0])
                avg_bw = float(values[-2]) / 8.0
                metric = f'{self.__support_rdma_commands[self._args.commands[cmd_idx]]}_{size}:'
                metric += self._get_metric_suffix()
                # Filter useless value in client output
                if metric not in metric_set:
                    metric_set.add(metric)
                    self._result.add_result(metric, avg_bw)
                    valid = True
            except BaseException:
                pass
        if valid is False:
            logger.error(
                'The result format is invalid - round: {}, benchmark: {}, raw output: {}.'.format(
                    self._curr_run_index, self._name, raw_output
                )
            )
            return False

        return True


BenchmarkRegistry.register_benchmark('rdma-loopback', RdmaLoopbackBenchmark)
