# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unified PyTorch deterministic training example for all supported models.

Deterministic metrics (loss, activation mean) are automatically stored in results.json
when --enable-determinism flag is enabled. Use --compare-log to compare against a reference run.

Commands to run:
Run A (generate reference):

python3 examples/benchmarks/pytorch_deterministic_example.py \
    --model <model_from_MODEL_CHOICES> --enable-determinism --deterministic-seed 42

This creates results-0.json with deterministic metrics.

Run B (compare against reference):

python3 examples/benchmarks/pytorch_deterministic_example.py \
    --model <model_from_MODEL_CHOICES> --enable-determinism --deterministic-seed 42 --compare-log results-0.json

Note: CUBLAS_WORKSPACE_CONFIG is now automatically set by the code when determinism is enabled.
"""

import argparse
import json
from pathlib import Path
from superbench.benchmarks import BenchmarkRegistry, Framework
from superbench.common.utils import logger

MODEL_CHOICES = [
    'bert-large',
    'gpt2-small',
    'llama2-7b',
    'mixtral-8x7b',
    'resnet101',
    'lstm',
]

DEFAULT_PARAMS = {
    'bert-large':
    '--batch_size 1 --seq_len 64 --num_warmup 1 --num_steps 200 --precision float32 '
    '--model_action train --check_frequency 20',
    'gpt2-small':
    '--batch_size 1 --num_steps 300 --num_warmup 1 --seq_len 128 --precision float32 '
    '--model_action train --check_frequency 20',
    'llama2-7b':
    '--batch_size 1 --num_steps 300 --num_warmup 1 --seq_len 512 --precision float32 --model_action train '
    '--check_frequency 20',
    'mixtral-8x7b':
    '--hidden_size=4096 --num_hidden_layers=32 --num_attention_heads=32 --intermediate_size=14336 '
    '--num_key_value_heads=8 --max_position_embeddings=32768 --router_aux_loss_coef=0.02 '
    '--check_frequency 20',
    'resnet101':
    '--batch_size 1 --precision float32 --num_warmup 1 --num_steps 120 --sample_count 8192 '
    '--pin_memory --model_action train --check_frequency 20',
    'lstm':
    '--batch_size 1 --num_steps 100 --num_warmup 2 --seq_len 64 --precision float16 '
    '--model_action train --check_frequency 30',
}


def main():
    """Main function for determinism example file."""
    parser = argparse.ArgumentParser(description='Unified PyTorch deterministic training example.')
    parser.add_argument('--model', type=str, choices=MODEL_CHOICES, required=True, help='Model to run.')
    parser.add_argument(
        '--enable-determinism',
        '--enable_determinism',
        action='store_true',
        help='Enable deterministic mode for reproducible results.',
    )
    parser.add_argument(
        '--compare-log',
        type=str,
        default=None,
        help='Path to reference results.json file for deterministic comparison.',
    )
    parser.add_argument(
        '--deterministic-seed',
        type=int,
        default=None,
        help='Seed for deterministic training.',
    )
    args = parser.parse_args()

    parameters = DEFAULT_PARAMS[args.model]
    if args.enable_determinism:
        parameters += ' --enable-determinism'
    if args.deterministic_seed is not None:
        parameters += f' --deterministic_seed {args.deterministic_seed}'
    if args.compare_log:
        parameters += f' --compare-log {args.compare_log}'

    context = BenchmarkRegistry.create_benchmark_context(args.model, parameters=parameters, framework=Framework.PYTORCH)
    benchmark = BenchmarkRegistry.launch_benchmark(context)
    logger.info(f'Benchmark finished. Return code: {benchmark.return_code}')

    # Save results to file for comparison
    if not args.compare_log:
        # Find next available results file name
        counter = 0
        while Path(f'results-{counter}.json').exists():
            counter += 1
        results_file = f'results-{counter}.json'

        # Parse benchmark results and create nested format like results-summary.json
        benchmark_results = json.loads(benchmark.serialized_result)

        # Create nested structure: raw_data -> benchmark_name -> metrics
        # Extract the benchmark name from the results (e.g., "pytorch-lstm")
        benchmark_name = benchmark_results.get('name', args.model)

        # Create results in the format expected by comparison logic
        nested_results = {
            'raw_data': {
                f'model-benchmarks:{args.model}/{benchmark_name}': benchmark_results.get('raw_data', {})
            }
        }

        # Write results to file
        with open(results_file, 'w') as f:
            json.dump(nested_results, f, indent=2)
        logger.info(f'Results saved to {results_file}')
        logger.info(f'To compare against this run, use: --compare-log {results_file}')
    else:
        logger.info(f'Comparison completed against {args.compare_log}')

    if hasattr(benchmark, '_model_run_metadata'):
        logger.info(f'Run metadata: {benchmark._model_run_metadata}')
    if hasattr(benchmark, '_model_run_periodic'):
        num_checkpoints = len(benchmark._model_run_periodic.get('step', []))
        logger.info(f'Periodic fingerprints collected at {num_checkpoints} checkpoints')


if __name__ == '__main__':
    main()
