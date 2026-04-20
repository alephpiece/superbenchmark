# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GPU topology utilities."""

import json
import re

from superbench.common.utils.process import run_command


def get_gpu_numa_map():
    """Get NUMA topology for all local GPUs.

    Returns:
        dict: GPU NUMA topology keyed by GPU id.
    """
    output = run_command('hy-smi --showtoponuma --json', quiet=True)
    if output.returncode != 0:
        raise RuntimeError('Failed to get GPU NUMA topology from hy-smi - message: {}'.format(output.stdout))

    try:
        hygon_topology = json.loads(output.stdout)
        gpu_numa_map = {}
        for card, card_topology in hygon_topology.items():
            match = re.fullmatch(r'card(\d+)', card)
            if not match:
                continue
            gpu_id = int(match.group(1))
            numa_node = card_topology['(Topology) Numa Node']
            numa_affinity = card_topology.get('(Topology) Numa Affinity', numa_node)
            int(numa_node)
            int(numa_affinity)
            gpu_numa_map[gpu_id] = {
                'numa_node': numa_node,
                'numa_affinity': numa_affinity,
            }
        if not gpu_numa_map:
            raise ValueError('no card topology found')
    except Exception as e:
        raise RuntimeError('Failed to parse GPU NUMA topology from hy-smi - message: {}'.format(e))

    return gpu_numa_map


def get_gpu_numa_affinity(gpu_id):
    """Get NUMA affinity for a GPU.

    Args:
        gpu_id (int): GPU id.

    Returns:
        str: GPU NUMA affinity.
    """
    try:
        gpu_id = int(gpu_id)
        return get_gpu_numa_map()[gpu_id]['numa_affinity']
    except Exception as e:
        raise RuntimeError('Failed to get GPU NUMA affinity - gpu_id: {}, message: {}'.format(gpu_id, e))
