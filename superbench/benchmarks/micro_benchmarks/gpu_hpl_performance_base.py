# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the GPU HPL benchmark base class."""

import os
from typing import Optional

from superbench.common.utils import logger
from superbench.benchmarks.micro_benchmarks import MicroBenchmarkWithInvoke


def _format_pmap(pmap):
    """Format HPL process mapping token used in the T/V field."""
    return 'R' if pmap == 0 else 'C'


def _format_fact(fact):
    """Format HPL panel factorization token used in the T/V field."""
    fact_tokens = {
        0: 'L',
        1: 'C',
        2: 'R',
    }
    return fact_tokens[fact]


def _format_transpose(value):
    """Format HPL transposed/no-transposed token used in the extended T/V field."""
    return 'T' if value == 0 else 'N'


def _format_equilibration(value):
    """Format HPL equilibration token used in the extended T/V field."""
    return 'N' if value == 0 else 'Y'


def format_hpl_mxp_tv(pmap, bcast):
    """Format the rocHPL-MxP T/V field from input parameters."""
    return f'W{_format_pmap(pmap)}{bcast}'


def format_hpl_tv(pmap, depth, bcast, rfact, ndiv, pfact, nbmin):
    """Format the rocHPL T/V field from input parameters."""
    return (
        f'W{_format_pmap(pmap)}'
        f'{depth}'
        f'{bcast}'
        f'{_format_fact(rfact)}'
        f'{ndiv}'
        f'{_format_fact(pfact)}'
        f'{nbmin}'
    )


def format_hpl_extended_tv(pmap, depth, bcast, rfact, ndiv, pfact, nbmin, l1, u, equilibration, alignment):
    """Format the rocHPL T/V field plus SuperBench variant suffix from input parameters."""
    return (
        f'{format_hpl_tv(pmap, depth, bcast, rfact, ndiv, pfact, nbmin)}'
        f'_{_format_transpose(l1)}'
        f'{_format_transpose(u)}'
        f'{_format_equilibration(equilibration)}'
        f'{alignment}'
    )


class GpuHplBenchmark(MicroBenchmarkWithInvoke):
    """The GPU HPL benchmark base class."""

    _default_bin_name: Optional[str] = None
    _default_dat_name: Optional[str] = None
    _default_out_name: Optional[str] = None
    _file_label: Optional[str] = None
    _default_n = 45312
    _default_nb = 384

    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        super().__init__(name, parameters)

        self._bin_name = self._default_bin_name
        self._dat_path = None
        self._out_path = None
        self._tv = None
        self._workload = None
        self._dat_file_name = None
        self._out_file_name = None

    def add_parser_arguments(self):
        """Add the specified arguments."""
        super().add_parser_arguments()

        self._parser.add_argument(
            '-P',
            '--P',
            type=int,
            default=1,
            required=False,
            help='Specific MPI grid size: the number of rows in MPI grid.',
        )
        self._parser.add_argument(
            '-Q',
            '--Q',
            type=int,
            default=1,
            required=False,
            help='Specific MPI grid size: the number of columns in MPI grid.',
        )
        self._parser.add_argument(
            '-p',
            '--p',
            type=int,
            required=False,
            help='Specific node-local MPI grid size: the number of rows in node-local MPI grid.',
        )
        self._parser.add_argument(
            '-q',
            '--q',
            type=int,
            required=False,
            help='Specific node-local MPI grid size: the number of columns in node-local MPI grid.',
        )
        self._parser.add_argument(
            '-N',
            '--N',
            type=int,
            default=self._default_n,
            required=False,
            help='Specific matrix size: the number of rows/columns in global matrix.',
        )
        self._parser.add_argument(
            '-NB',
            '--NB',
            type=int,
            default=self._default_nb,
            required=False,
            help='Specific panel size: the number of rows/columns in panels.',
        )
        self._parser.add_argument(
            '--warmup',
            type=int,
            default=0,
            required=False,
            help='Number of warmup runs to exclude from result aggregation.',
        )
        self._parser.add_argument(
            '--iterations',
            type=int,
            default=1,
            required=False,
            help='Number of measurement runs to include in result aggregation.',
        )
        self._parser.add_argument(
            '--reduce-op',
            dest='reduce_op',
            type=str,
            default='max',
            choices=['mean', 'median', 'max', 'min'],
            required=False,
            help='Reduce operator for aggregating measurement runs by FLOPS.',
        )
        self._parser.add_argument(
            '--PMAP',
            type=int,
            default=1,
            choices=[0, 1],
            required=False,
            help='Process mapping: 0 for row-major, 1 for column-major.',
        )
        self._parser.add_argument(
            '--BCAST',
            type=int,
            default=0,
            choices=[0, 1, 2, 3, 4, 5],
            required=False,
            help='Broadcast topology: 0 for 1rg, 1 for 1rM, 2 for 2rg, 3 for 2rM, 4 for Lng, 5 for LnM.',
        )
        self._parser.add_argument(
            '--threshold',
            type=float,
            default=16.0,
            required=False,
            help='Residual check threshold.',
        )

        self._add_variant_parser_arguments()

    def _add_variant_parser_arguments(self):
        """Add benchmark variant-specific arguments."""
        pass

    def _preprocess(self):
        """Preprocess/preparation operations before benchmarking."""
        if not super()._preprocess():
            return False

        if self._args.warmup < 0:
            logger.error('warmup should be non-negative, while {} is set.'.format(self._args.warmup))
            return False
        if self._args.iterations <= 0:
            logger.error('iterations should be positive, while {} is set.'.format(self._args.iterations))
            return False

        self._tv = self._format_tv()
        self._workload = self._format_workload()
        file_prefix = self._format_file_prefix()
        self._dat_file_name = f'{file_prefix}.dat'
        self._out_file_name = f'{file_prefix}.out'
        self._dat_path = os.path.join(self._args.bin_dir, self._dat_file_name)
        self._out_path = os.path.join(self._args.bin_dir, self._out_file_name)

        with open(self._dat_path, 'w') as dat_file:
            dat_file.write(self._format_dat_content())

        bin_path = os.path.join(self._args.bin_dir, self._bin_name)
        command = (
            f'{bin_path}'
            f' -P {self._args.P}'
            f' -Q {self._args.Q}'
            f' --it {self._args.warmup + self._args.iterations}'
            f' -i {self._dat_file_name}'
        )
        if self._args.p is not None:
            command += f' -p {self._args.p}'
        if self._args.q is not None:
            command += f' -q {self._args.q}'

        self._commands = [command]
        return True

    def _process_raw_result(self, cmd_idx, raw_output):
        """Parse HPL stdout and generated output file."""
        raise NotImplementedError

    def _format_tv(self):
        """Format the expected T/V field from benchmark input arguments."""
        raise NotImplementedError

    def _format_workload(self):
        """Format the metric workload suffix from benchmark input arguments."""
        return f'{self._tv}_P{self._args.P}_Q{self._args.Q}_N{self._args.N}_NB{self._args.NB}'

    def _format_file_prefix(self):
        """Format generated HPL.dat/HPL.out file prefix."""
        return f'{self._file_label or self._name}-{self._workload}'

    def _format_dat_content(self):
        """Format generated HPL.dat content."""
        raise NotImplementedError
