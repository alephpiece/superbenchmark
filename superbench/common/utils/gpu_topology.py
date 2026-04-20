# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""GPU topology utilities."""


def get_gpu_numa_node_command(proc_rank):
    """Get command substitution for GPU NUMA node according to local process rank.

    Args:
        proc_rank (int): Local process rank.

    Returns:
        str: Shell command substitution for GPU NUMA node.
    """
    parse_command = (
        'python3 -c "import json,sys;'
        'print(json.load(sys.stdin)[\\"card{}\\"][\\"(Topology) Numa Node\\"])"'.format(proc_rank)
    )
    return '$(hy-smi --showtoponuma --json | {})'.format(parse_command)
