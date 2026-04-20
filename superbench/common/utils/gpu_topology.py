# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GPU topology utilities."""

import json

from superbench.common.utils.process import run_command


def get_gpu_numa_node(proc_rank):
    """Get GPU NUMA node according to local process rank.

    Args:
        proc_rank (int): Local process rank.

    Returns:
        str: GPU NUMA node.
    """
    output = run_command('hy-smi --showtoponuma --json', quiet=True)
    if output.returncode != 0:
        raise RuntimeError('Failed to get GPU NUMA node from hy-smi - message: {}'.format(output.stdout))

    try:
        topology = json.loads(output.stdout)
        numa_node = topology['card{}'.format(proc_rank)]['(Topology) Numa Node']
        int(numa_node)
    except Exception as e:
        raise RuntimeError(
            'Failed to parse GPU NUMA node from hy-smi - proc_rank: {}, message: {}'.format(proc_rank, e)
        )

    return numa_node
