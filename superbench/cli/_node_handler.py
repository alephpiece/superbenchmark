# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SuperBench CLI node subgroup command handler."""

from pathlib import Path
import json

from knack.util import CLIError

from superbench.tools import SystemInfo
from superbench.common.utils import create_sb_output_dir
from superbench.common.utils.gpu_topology import get_gpu_numa_node


def info_command_handler(output_dir=None):
    """Get node hardware info.

    Args:
        output_dir (str): Output directory.

    Returns:
        dict: node info.
    """
    try:
        info = SystemInfo().get_all()
        output_dir = create_sb_output_dir(output_dir)
        output_dir_path = Path(output_dir)
        with open(output_dir_path / 'sys_info.json', 'w') as f:
            json.dump(info, f)
    except Exception as ex:
        raise RuntimeError('Failed to get node info.') from ex
    return info


def topo_command_handler(get=None, proc_rank=None):
    """Get node topology information.

    Args:
        get (str): Topology field to get.
        proc_rank (int): Local process rank.
    """
    if get != 'gpu-numa-node':
        raise CLIError('Unsupported topology field: {}.'.format(get))
    if proc_rank is None:
        raise CLIError('--proc-rank is required for gpu-numa-node.')

    print(get_gpu_numa_node(proc_rank))
