# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utility functions for deterministic model training and validation."""

import json


def build_model_metadata(name, precision, args, extra_keys=None):
    """Build metadata dictionary for deterministic model runs.

    Args:
        name (str): Model name.
        precision: Model precision (enum or string).
        args: Parsed arguments object.
        extra_keys (list): Additional argument keys to include in metadata.

    Returns:
        dict: Metadata dictionary with model configuration.
    """
    metadata = {
        'model_name': name,
        'precision': (precision.value if hasattr(precision, 'value') else str(precision)),
        'seed': getattr(args, 'deterministic_seed', None),
        'deterministic_seed': getattr(args, 'deterministic_seed', None),
        'batch_size': getattr(args, 'batch_size', None),
        'seq_len': getattr(args, 'seq_len', None),
        'num_steps': getattr(args, 'num_steps', None),
        'num_warmup': getattr(args, 'num_warmup', None),
        'check_frequency': getattr(args, 'check_frequency', None),
        'num_classes': getattr(args, 'num_classes', None),
    }

    # Add common model architecture keys
    keys = [
        'hidden_size',
        'num_hidden_layers',
        'num_attention_heads',
        'intermediate_size',
        'input_size',
        'num_layers',
        'bidirectional',
    ]
    if extra_keys:
        keys += extra_keys

    for key in keys:
        metadata[key] = getattr(args, key, None)

    return metadata


def record_step_loss(loss, curr_step, losses_list, logger=None):
    """Record per-step loss value for determinism tracking.

    Args:
        loss: Loss tensor or float value.
        curr_step (int): Current training step.
        losses_list (list): List to append loss values to.
        logger: Optional logger for warnings.

    Returns:
        float: Converted loss value, or None if conversion failed.
    """
    try:
        v = float(loss.detach().item()) if hasattr(loss, 'detach') else float(loss)
        losses_list.append(v)
        return v
    except Exception:
        if logger:
            logger.info(f'Unable to convert loss to float at step {curr_step}')
        losses_list.append(None)
        return None


def record_periodic_fingerprint(
    curr_step, loss_value, logits, periodic_dict, check_frequency, enable_determinism, logger=None
):
    """Record periodic fingerprints (loss and activation mean) for deterministic runs.

    Args:
        curr_step (int): Current training step.
        loss_value: Pre-converted loss float value (or None).
        logits: Logits tensor for activation fingerprint.
        periodic_dict (dict): Dictionary to store periodic data ('loss', 'act_mean', 'step').
        check_frequency (int): Frequency for fingerprint logging.
        enable_determinism (bool): Whether determinism is enabled.
        logger: Optional logger for info/warnings.
    """
    if not enable_determinism or (curr_step % check_frequency != 0):
        return

    # 1) Loss fingerprint (only at fingerprinting frequency)
    try:
        if 'loss' in periodic_dict and isinstance(periodic_dict['loss'], list):
            periodic_dict['loss'].append(loss_value if loss_value is not None else None)
        else:
            periodic_dict['loss'] = [loss_value if loss_value is not None else None]

        if logger:
            logger.info(f'Loss at step {curr_step}: {loss_value}')
        periodic_dict.setdefault('step', []).append(curr_step)
    except Exception:
        if logger:
            logger.warning(f'Unable to log loss at curr_step {curr_step}')

    # 2) Activation fingerprint: mean over logits for sample 0
    try:
        if logits is not None:
            act_mean = (
                float(logits[0].detach().float().mean().item()) if hasattr(logits[0], 'detach') else float(logits[0])
            )
            if logger:
                logger.info(f'ActMean at step {curr_step}: {act_mean}')
            periodic_dict.setdefault('act_mean', []).append(act_mean)
        else:
            # Keep lists aligned by appending None when activation not available
            periodic_dict.setdefault('act_mean', []).append(None)
    except Exception:
        if logger:
            logger.warning(f'Unable to log act_mean at curr_step {curr_step}')
        periodic_dict.setdefault('act_mean', []).append(None)


def load_reference_results(filepath, benchmark_name, rank=None, logger=None):
    """Load reference results file and extract raw_data for a specific benchmark.

    Args:
        filepath (str): Path to reference results JSON file.
        benchmark_name (str): Name of the benchmark to extract.
        rank (int): Optional rank number for distributed training.
        logger: Optional logger for warnings.

    Returns:
        tuple: (ref_raw_data dict, ref_metadata dict) or (None, None) on error.

    Raises:
        FileNotFoundError: If reference file doesn't exist.
        ValueError: If reference file is invalid or missing data.
    """
    try:
        with open(filepath, 'r') as f:
            ref_results = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f'Reference results file not found: {filepath}. '
            f'Make sure you have run the benchmark with --enable-determinism first to generate reference results.'
        )
    except json.JSONDecodeError as e:
        raise ValueError(f'Invalid JSON in reference results file {filepath}: {e}')

    # Get raw_data section
    if 'raw_data' not in ref_results:
        raise ValueError(f'Reference file {filepath} does not contain "raw_data" section')

    ref_raw_data_section = ref_results['raw_data']

    # Find benchmark in nested format
    ref_raw_data = None
    for bm_name in ref_raw_data_section:
        if benchmark_name in bm_name:
            ref_raw_data = ref_raw_data_section[bm_name]
            break

    if ref_raw_data is None:
        raise ValueError(
            f'Reference file does not contain raw_data for benchmark matching "{benchmark_name}". '
            f'Available benchmarks: {list(ref_raw_data_section.keys())}'
        )

    # Extract metadata
    ref_metadata = None
    if rank is not None:
        metadata_key = f'metadata_rank{rank}'
    else:
        metadata_key = 'metadata'

    if metadata_key in ref_raw_data:
        metadata_list = ref_raw_data[metadata_key]
        ref_metadata = _extract_metadata_from_raw_data(metadata_list)
    elif 'metadata_rank0' in ref_raw_data:
        # Fallback to rank 0 metadata
        metadata_list = ref_raw_data['metadata_rank0']
        ref_metadata = _extract_metadata_from_raw_data(metadata_list)

    return ref_raw_data, ref_metadata


def _extract_metadata_from_raw_data(metadata_list):
    """Extract metadata dict from raw_data list format.

    Args:
        metadata_list: Metadata in raw_data format (list of lists or list of dicts).

    Returns:
        dict: Extracted metadata, or None if extraction failed.
    """
    if isinstance(metadata_list, list) and len(metadata_list) > 0:
        first_item = metadata_list[0]
        if isinstance(first_item, dict):
            return first_item
        elif isinstance(first_item, list) and len(first_item) > 0 and isinstance(first_item[0], dict):
            return first_item[0]
    elif isinstance(metadata_list, dict):
        return metadata_list
    return None


def compare_raw_data_metrics(curr_raw_data, ref_raw_data, rank=None, logger=None):
    """Compare current and reference raw_data metrics for determinism validation.

    Args:
        curr_raw_data (dict): Current run's raw_data.
        ref_raw_data (dict): Reference run's raw_data.
        rank (int): Optional rank number for distributed training.
        logger: Optional logger for debug messages.

    Returns:
        list: List of mismatch descriptions, empty if all match.
    """
    mismatches = []

    # Determine metric prefix
    if rank is not None:
        metric_prefix = f'deterministic_loss_rank{rank}'
    else:
        metric_prefix = 'deterministic_loss'

    # Check if deterministic metrics exist in reference
    if metric_prefix not in ref_raw_data:
        raise ValueError(
            f'Reference results do not contain deterministic metrics ({metric_prefix}) in raw_data. '
            f'Make sure the reference was run with --enable-determinism flag.'
        )

    # Compare deterministic raw data
    for key in curr_raw_data:
        if key.startswith('deterministic_') and key in ref_raw_data:
            curr_val = curr_raw_data[key]
            ref_val = ref_raw_data[key]

            if isinstance(curr_val, list) and isinstance(ref_val, list):
                # Raw data is list of lists for multiple runs
                if len(curr_val) != len(ref_val):
                    mismatches.append(f'{key}: run count mismatch ({len(curr_val)} vs {len(ref_val)})')
                    continue

                for run_idx in range(len(curr_val)):
                    curr_run = curr_val[run_idx]
                    ref_run = ref_val[run_idx]

                    if len(curr_run) != len(ref_run):
                        mismatches.append(
                            f'{key}[run {run_idx}]: checkpoint count mismatch ({len(curr_run)} vs {len(ref_run)})'
                        )
                        continue

                    # Compare each checkpoint value for exact equality
                    for step_idx, (curr_step_val, ref_step_val) in enumerate(zip(curr_run, ref_run)):
                        if logger:
                            logger.debug(f'{key}[{run_idx},{step_idx}]: {curr_step_val} vs {ref_step_val}')
                        if curr_step_val != ref_step_val:
                            if isinstance(curr_step_val, (int, float)) and isinstance(ref_step_val, (int, float)):
                                diff_val = abs(curr_step_val - ref_step_val)
                                mismatches.append(
                                    f'{key}[run {run_idx}, checkpoint {step_idx}]: '
                                    f'{repr(curr_step_val)} vs {repr(ref_step_val)} (diff: {diff_val})'
                                )
                            else:
                                mismatches.append(
                                    f'{key}[run {run_idx}, checkpoint {step_idx}]: '
                                    f'{repr(curr_step_val)} vs {repr(ref_step_val)}'
                                )

    return mismatches


def apply_metadata_overrides(args, ref_metadata, logger=None):
    """Apply reference metadata overrides to current args for reproducibility.

    Args:
        args: Parsed arguments object to modify.
        ref_metadata (dict): Reference metadata with configuration.
        logger: Optional logger for info messages.

    Returns:
        int: Number of parameters overridden.
    """
    if not ref_metadata:
        if logger:
            logger.warning('No metadata provided for override')
        return 0

    override_params = [
        'batch_size', 'seq_len', 'hidden_size', 'num_steps', 'num_warmup', 'check_frequency', 'num_classes',
        'num_layers', 'num_hidden_layers', 'num_attention_heads', 'intermediate_size', 'input_size', 'bidirectional',
        'seed', 'precision', 'deterministic_seed'
    ]

    overridden_count = 0
    for param in override_params:
        if param in ref_metadata and hasattr(args, param):
            ref_value = ref_metadata[param]
            curr_value = getattr(args, param)

            # Handle precision specially - it must be a list
            if param == 'precision':
                if isinstance(ref_value, str):
                    # Convert string to Precision enum and wrap in list
                    from superbench.benchmarks.context import Precision
                    ref_value = [Precision(ref_value)]
                elif isinstance(ref_value, list):
                    # Ensure list items are Precision enums
                    from superbench.benchmarks.context import Precision
                    ref_value = [Precision(v) if isinstance(v, str) else v for v in ref_value]

            if ref_value != curr_value:
                if logger:
                    logger.info(f'Overriding {param} from {curr_value} to {ref_value} (from reference metadata)')
                setattr(args, param, ref_value)
                overridden_count += 1

    return overridden_count
