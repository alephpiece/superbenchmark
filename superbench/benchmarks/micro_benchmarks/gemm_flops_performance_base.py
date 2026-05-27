# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the FLOPs performance benchmark base class."""

import itertools

from superbench.common.utils import logger
from superbench.benchmarks import ReturnCode
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


def mrange(start, stop=-1, factor=2, symbol='x'):
    """Range constructor with multiplication or addition factor."""
    if stop == -1:
        yield start
        return

    if symbol == 'x':
        while True:
            yield start
            start *= factor
            if start > stop or start == 0 or factor < 2:
                break
    elif symbol == '+':
        while True:
            yield start
            start += factor
            if start > stop or start == 0 or factor < 1:
                break
    else:
        raise ValueError(f'Invalid symbol {symbol}.')


def validate_mrange(string):
    """Validate mrange string in format start[[:stop]:factor]."""
    nums = string.split(':')
    if len(nums) > 3:
        return False

    if len(nums) < 3:
        return all(x.isdigit() for x in nums)
    return nums[0].isdigit() and nums[1].isdigit() and (nums[2].lstrip('+').isdigit() or nums[2].lstrip('x').isdigit())


class GemmFlopsBenchmark(MicroBenchmarkWithInvoke):
    """The GEMM FLOPs performance benchmark base class."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._support_precisions = [
            'fp64', 'fp32', 'fp16', 'fp64_tc', 'tf32_tc', 'bf16_tc', 'fp16_tc', 'int8_tc', 'int4_tc'
        ]
        self._precision_need_to_run = list()
        self._shapes_to_run = list()
        self._metric_map = {
            'fp64': 'fp64_flops',
            'fp32': 'fp32_flops',
            'fp16': 'fp16_flops',
            'fp64_tc': 'fp64_tc_flops',
            'tf32_tc': 'tf32_tc_flops',
            'bf16_tc': 'bf16_tc_flops',
            'fp16_tc': 'fp16_tc_flops',
            'int8_tc': 'int8_tc_iops',
            'int4_tc': 'int4_tc_iops',
            'fp32_xdlops': 'fp32_xdlops_flops',
            'fp16_xdlops': 'fp16_xdlops_flops',
            'bf16_xdlops': 'bf16_xdlops_flops',
            'int8_xdlops': 'int8_xdlops_iops'
        }

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '--num_warmup',
            type=int,
            default=2,
            required=False,
            help='The number of warmup step.',
        )
        self._parser.add_argument(
            '--n',
            type=int,
            default=16384,
            required=False,
            help='The N dim of matmul (N, K) * (K, M).',
        )
        self._parser.add_argument(
            '--k',
            type=int,
            default=16384,
            required=False,
            help='The K dim of matmul (N, K) * (K, M).',
        )
        self._parser.add_argument(
            '--m',
            type=int,
            default=16384,
            required=False,
            help='The M dim of matmul (N, K) * (K, M).',
        )
        self._parser.add_argument(
            '--shapes',
            type=str,
            nargs='+',
            default=list(),
            help='Shapes in m,n,k format. Support format start:stop:factor, e.g., 4096:32768:2.',
        )
        self._parser.add_argument(
            '--precision',
            type=str,
            nargs='+',
            default=list(),
            help='Precision for benchmarking. E.g. {}.'.format(' '.join(self._support_precisions)),
        )

    def _preprocess(self):
        """Preprocess/preparation operations before the benchmarking.

        Return:
            True if _preprocess() succeed.
        """
        if not super()._preprocess():
            return False

        if len(self._args.precision) == 0:
            self._precision_need_to_run = self._support_precisions
        else:
            self._args.precision = [p.lower() for p in self._args.precision]
            for p in self._args.precision:
                if p not in self._support_precisions:
                    logger.warning(
                        'Unsupported precision - benchmark: {}, precision: {}, expected: {}.'.format(
                            self._name, p, self._support_precisions
                        )
                    )
                else:
                    self._precision_need_to_run.append(p)

        if len(self._precision_need_to_run) == 0:
            self._result.set_return_code(ReturnCode.NO_SUPPORTED_PRECISION)
            return False

        shapes = self._args.shapes or [f'{self._args.m},{self._args.n},{self._args.k}']
        for shape in shapes:
            shape_list = shape.replace(',', ' ').split()
            if len(shape_list) != 3 or not all(validate_mrange(x) for x in shape_list):
                logger.error(f'Invalid shape - benchmark: {self._name}, shape: {shape}.')
                return False

            for m, n, k in itertools.product(
                *map(
                    lambda dim: mrange(
                        *map(lambda value: int(value.lstrip('+').lstrip('x')), dim.split(':')),
                        symbol=dim.split(':')[2][0]
                        if len(dim.split(':')) == 3 and any([operator in dim for operator in ['+', 'x']]) else 'x'
                    ), shape_list
                )
            ):
                self._shapes_to_run.append((m, n, k))

        return True

    def _get_metric_name(self, precision, m, n, k):
        """Build metric name with precision and GEMM shape."""
        metric = self._metric_map[precision]
        if metric.endswith('_flops'):
            return f'{metric[:-len("_flops")]}_m{m}_n{n}_k{k}_flops'
        if metric.endswith('_iops'):
            return f'{metric[:-len("_iops")]}_m{m}_n{n}_k{k}_iops'
        return f'{metric}_m{m}_n{n}_k{k}'
