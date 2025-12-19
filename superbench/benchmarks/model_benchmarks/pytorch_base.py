# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Module of the Pytorch model-benchmark base class."""

import os
from datetime import timedelta
import random
import time

import torch
import transformers
try:
    import transformer_engine.pytorch as te
except ImportError:
    te = None
from torch.backends.cuda import sdp_kernel
from torch.distributed import TCPStore, PrefixStore
from torch.utils.data import DataLoader

from superbench.common.utils import logger
from superbench.common import model_log_utils
from superbench.benchmarks import (
    Framework,
    ReturnCode,
    DistributedBackend,
    DistributedImpl,
)
from superbench.benchmarks.model_benchmarks.model_base import Optimizer, ModelBenchmark


class PytorchBase(ModelBenchmark):
    """The base class of Pytorch model benchmarks."""
    def __init__(self, name, parameters=''):
        """Constructor.

        Args:
            name (str): benchmark name.
            parameters (str): benchmark parameters.
        """
        # Set CUBLAS_WORKSPACE_CONFIG early, before parent init which might parse args
        # This ensures it's set before any CUDA operations if determinism is enabled
        if 'enable-determinism' in parameters or 'enable_determinism' in parameters:
            os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

        super().__init__(name, parameters)

        self._framework = Framework.PYTORCH
        torch.backends.cudnn.benchmark = True

        self._model_run_metadata = {}
        self._model_run_losses = []
        self._model_run_periodic = {}

    def _judge_gpu_availability(self):
        """Judge GPUs' availability according to arguments and running environment."""
        self._gpu_available = not self._args.no_gpu and torch.cuda.is_available()

    def _enable_deterministic_training(self):
        """Enable deterministic training settings for reproducible results."""
        # Set CUBLAS_WORKSPACE_CONFIG (should already be set in __init__, but ensure it's set as backup)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

        if hasattr(self._args, 'deterministic_seed'):
            torch.manual_seed(self._args.deterministic_seed)
            random.seed(self._args.deterministic_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self._args.deterministic_seed)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Disable TF32 to remove potential numerical variability
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            logger.warning('Failed to disable TF32 in cuda matmul')

        try:
            torch.backends.cudnn.allow_tf32 = False
        except Exception:
            logger.warning('Failed to disable TF32 in cuDNN')

        # Force Scaled Dot-Product Attention to use deterministic math kernel
        try:
            sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
        except Exception:
            logger.warning('SDP kernel not available')
            # Older PyTorch versions may not expose sdp_kernel; ignore in that case

    def _assign_model_run_metadata(self, precision, extra_keys=None):
        """Assign model_run_metadata for determinism fingerprinting/logging.

        Args:
            precision: Model precision (can be enum or string).
            extra_keys: List of additional argument keys to include in metadata.

        Returns:
            None
        """
        self._model_run_metadata = model_log_utils.build_model_metadata(self._name, precision, self._args, extra_keys)
        return None

    def record_determinism_fingerprint(self, curr_step, loss, logits, periodic, check_frequency):
        """Centralized logic for recording per-step loss and periodic fingerprints for deterministic runs.

        Args:
            curr_step (int): Current training step.
            loss (torch.Tensor or float): Loss value for this step.
            logits (torch.Tensor or float): Logits output for this step (sample 0).
            periodic (dict): Dictionary to store periodic fingerprints ('loss', 'act_mean', 'step').
            check_frequency (int): Frequency for fingerprint logging.
        """
        # Record per-step loss for determinism checks
        loss_value = model_log_utils.record_step_loss(loss, curr_step, self._model_run_losses, logger)

        # Record periodic fingerprint (loss and activation mean)
        model_log_utils.record_periodic_fingerprint(
            curr_step, loss_value, logits, periodic, check_frequency, getattr(self._args, 'enable_determinism', False),
            logger
        )

    def _finalize_periodic_logging(self, periodic, info_key='loss'):
        """Finalize periodic logging and return info dict for training step."""
        info = {info_key: periodic.get(info_key, [])}
        self._model_run_losses = list(periodic.get(info_key, []))
        self._model_run_periodic = dict(periodic)
        return info

    def add_parser_arguments(self):
        """Add PyTorch model benchmark-specific arguments to the argument parser."""
        super().add_parser_arguments()
        self._parser.add_argument(
            '--compare-log',
            '--compare_log',
            dest='compare_log',
            type=str,
            default=None,
            help='Path to reference results.json file for deterministic comparison.',
        )
        self._parser.add_argument(
            '--deterministic_seed',
            '--deterministic-seed',
            type=int,
            default=42,
            required=False,
            help='Random seed for deterministic training.',
        )
        self._parser.add_argument(
            '--enable-determinism',
            '--enable_determinism',
            action='store_true',
            default=False,
            help='Enable deterministic training for reproducible results.',
        )
        self._parser.add_argument(
            '--generate_log',
            '--generate-log',
            action='store_true',
            default=False,
            help='Generate consolidated deterministic reference results (stores all ranks in results-summary).',
        )
        self._parser.add_argument(
            '--check_frequency',
            '--check-frequency',
            type=int,
            default=100,
            required=False,
            help='How often (in steps) to run lightweight periodic checks/logs and evaluate early-stop conditions.',
        )

    def _post_run_model_log(self):
        """Add deterministic metrics to results and optionally compare with reference results.

        Deterministic metrics (loss, activation mean) are stored in the results file alongside
        other benchmark metrics. When --compare-log is specified, loads the reference results
        file and compares deterministic metrics per-rank.
        """
        # Add deterministic metrics to result system (all ranks add their own metrics)
        if getattr(self._args, 'enable_determinism', False):
            self._add_deterministic_metrics_to_result()

            # Consolidate results from all ranks to rank 0 for complete results-summary
            # This is needed whether generating or comparing logs
            self._save_consolidated_deterministic_results()

            # Compare with reference results if requested
            if getattr(self._args, 'compare_log', None):
                self._compare_deterministic_results()

    def _add_deterministic_metrics_to_result(self):
        """Add deterministic fingerprints and losses to the benchmark result system.

        This makes deterministic metrics visible in results-summary.json alongside
        other benchmark metrics. In distributed training, metrics include rank information.
        """
        # Add periodic fingerprints (loss, activation mean) to results
        if self._model_run_periodic:
            for key, values in self._model_run_periodic.items():
                if isinstance(values, list) and values:
                    # Include rank in metric name for distributed training
                    if self._global_rank is not None:
                        metric_name = f'deterministic_{key}_rank{self._global_rank}'
                    else:
                        metric_name = f'deterministic_{key}'

                    # Add raw data (all values at each checkpoint)
                    self._result.add_raw_data(metric_name, values, self._args.log_raw_data)
                    # Add summarized result (mean of checkpointed values)
                    import statistics
                    self._result.add_result(metric_name, statistics.mean([v for v in values if v is not None]))

        # Add count of deterministic checks performed
        if self._model_run_periodic.get('step'):
            if self._global_rank is not None:
                metric_name = f'deterministic_check_count_rank{self._global_rank}'
            else:
                metric_name = 'deterministic_check_count'
            self._result.add_result(metric_name, len(self._model_run_periodic['step']))

        # Save metadata for configuration reproducibility
        if self._model_run_metadata:
            if self._global_rank is not None:
                metric_name = f'metadata_rank{self._global_rank}'
            else:
                metric_name = 'metadata'
            # Use False for log_raw_data to save in result object, not log file
            self._result.add_raw_data(metric_name, self._model_run_metadata, False)

    def _save_consolidated_deterministic_results(self):
        """Gather deterministic data from all ranks and save to results-summary (rank 0 only).

        In distributed training, all ranks send their raw_data to rank 0, which consolidates
        and adds it to the result system. This allows all ranks' checkpoint data to appear
        in the standard results-summary files.
        """
        import torch.distributed as dist

        # In distributed mode, gather all ranks' data to rank 0
        if self._args.distributed_impl == DistributedImpl.DDP:
            # Serialize current rank's raw_data
            raw_data_to_send = {}
            for key in self._result.raw_data:
                if key.startswith('deterministic_'):
                    raw_data_to_send[key] = self._result.raw_data[key]

            # Gather all ranks' data to rank 0
            if self._global_rank == 0:
                # Rank 0 collects data from all ranks
                all_ranks_data = [None] * dist.get_world_size()
                dist.gather_object(raw_data_to_send, all_ranks_data, dst=0)

                # Add all ranks' raw_data to rank 0's result (which becomes results-summary)
                for rank_idx, rank_data in enumerate(all_ranks_data):
                    if rank_data:
                        for key, value in rank_data.items():
                            # Add to rank 0's result raw_data if not already present
                            if key not in self._result.raw_data:
                                self._result.raw_data[key] = value

                logger.info(f'Rank 0: Consolidated deterministic results from {dist.get_world_size()} ranks')
            else:
                # Other ranks send their data to rank 0
                dist.gather_object(raw_data_to_send, None, dst=0)
        else:
            # Non-distributed: data already in result, nothing to consolidate
            logger.info('Deterministic results stored in results')

    def _compare_deterministic_results(self):
        """Compare current deterministic metrics with reference results file.

        Loads the reference results.json file and compares deterministic metrics
        (loss, activation mean) per-rank to verify reproducibility.
        """
        import torch.distributed as dist

        compare_log_path = self._args.compare_log
        rank = self._global_rank if self._global_rank is not None else 0
        logger.info(f'Rank {rank}: Loading reference results from {compare_log_path}')

        # Track if this rank detected any failure
        has_failure = False
        failure_msg = ''

        try:
            # Load reference results and extract raw_data
            ref_raw_data, _ = model_log_utils.load_reference_results(
                compare_log_path, self._name, self._global_rank, logger
            )

            # Compare metrics
            curr_raw_data = self._result.raw_data
            mismatches = model_log_utils.compare_raw_data_metrics(
                curr_raw_data, ref_raw_data, self._global_rank, logger
            )

            if mismatches:
                has_failure = True
                failure_msg = (
                    f'Rank {self._global_rank if self._global_rank is not None else 0}: '
                    f'Determinism check FAILED. Mismatched metrics:\n' + '\n'.join(mismatches)
                )
        except (FileNotFoundError, ValueError) as e:
            has_failure = True
            failure_msg = str(e)

        # Synchronize failure status across all ranks in distributed mode
        if self._args.distributed_impl == DistributedImpl.DDP:
            # Convert failure status to tensor for all_reduce
            import torch
            failure_tensor = torch.tensor([1 if has_failure else 0], dtype=torch.int32, device='cuda')
            dist.all_reduce(failure_tensor, op=dist.ReduceOp.MAX)

            # If any rank failed, all ranks should fail
            if failure_tensor.item() > 0:
                if has_failure:
                    # This rank detected the failure
                    logger.error(failure_msg)
                    raise RuntimeError(failure_msg)
                else:
                    # Another rank detected failure, fail together
                    error_msg = f'Rank {self._global_rank}: Determinism check FAILED on another rank'
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
        elif has_failure:
            # Non-distributed mode, just raise
            logger.error(failure_msg)
            raise RuntimeError(failure_msg)

        rank = self._global_rank if self._global_rank is not None else 0
        logger.info(f'Rank {rank}: Determinism check PASSED - all checkpoints match')

    def _preprocess(self):
        """Preprocess and apply PyTorch-specific defaults."""
        preprocess_ok = super()._preprocess()
        if not preprocess_ok:
            return False
        # Deterministic setup is handled centrally in set_deterministic_seed() which
        # is invoked earlier in the model-base preprocess before dataset creation.
        if getattr(self._args, 'enable_determinism', False):
            self._handle_deterministic_log_options()
        return True

    def set_deterministic_seed(self):
        """Set deterministic RNGs centrally for PyTorch benchmarks.

        This will set the seeds and deterministic flags prior to dataset generation
        so per-model dataset generation is reproducible without each model needing
        to call torch.manual_seed().
        """
        if getattr(self._args, 'enable_determinism', False):
            try:
                self._enable_deterministic_training()
            except Exception:
                logger.info('Failed to enable deterministic training in centralized preprocess')

    def _handle_deterministic_log_options(self):
        """Handle deterministic log options.

        In deterministic mode, metrics are automatically added to the results file.
        The --compare-log option can be used to compare against a previous results file.

        If compare-log is provided, load metadata from reference file and override current configuration
        to ensure exact reproducibility.
        """
        if self._args.compare_log:
            try:
                # Load reference metadata
                _, ref_metadata = model_log_utils.load_reference_results(
                    self._args.compare_log, self._name, self._global_rank, logger
                )

                if ref_metadata:
                    # Apply metadata overrides
                    overridden = model_log_utils.apply_metadata_overrides(self._args, ref_metadata, logger)
                    if overridden == 0:
                        logger.info('No parameters needed to be overridden from reference metadata')
                else:
                    logger.warning(
                        f'No metadata found in reference file {self._args.compare_log}. '
                        'Cannot verify configuration matches reference run.'
                    )
            except Exception as e:
                logger.warning(f'Failed to load metadata from reference file {self._args.compare_log}: {e}')

    def _set_force_fp32(self):
        """Set the config that controls whether full float32 precision will be used.

        On Ampere or newer GPUs, pytorch and tensorflow will use TF32 instead of FP32 by default.
        We can disable TF32 execution by setting force_fp32 as True.
        """
        torch.backends.cuda.matmul.allow_tf32 = not self._args.force_fp32
        torch.backends.cudnn.allow_tf32 = not self._args.force_fp32

    @torch.no_grad()
    def _to_te_model(self, model):
        """Convert the input model to Transformer Engine model.

        Replace all Linear/LayerNorm layers.
        Modified based on Huggingface's utils `accelerate.accelerator.convert_model`, reference:
        https://github.com/huggingface/accelerate/blob/v0.17.1/src/accelerate/utils/transformer_engine.py#L24

        Args:
            model (torch.nn.Module): Torch model.
        """
        if not te:
            return
        for name, m in model.named_children():
            if isinstance(m, torch.nn.Linear):
                # check 16-byte alignment
                if any(p % 16 != 0 for p in m.weight.shape):
                    return
                te_m = te.Linear(m.in_features, m.out_features, bias=(m.bias is not None), params_dtype=m.weight.dtype)
                te_m.weight.copy_(m.weight)
                if m.bias is not None:
                    te_m.bias.copy_(m.bias)
                setattr(model, name, te_m)
            elif isinstance(m, torch.nn.LayerNorm):
                te_m = te.LayerNorm(m.normalized_shape[0], eps=m.eps, params_dtype=m.weight.dtype)
                if hasattr(te_m, 'weight'):
                    te_m.weight.copy_(m.weight)
                    te_m.bias.copy_(m.bias)
                else:
                    te_m.layer_norm_weight.copy_(m.weight)
                    te_m.layer_norm_bias.copy_(m.bias)
                setattr(model, name, te_m)
            else:
                self._to_te_model(m)

    def _init_distributed_setting(self):
        """Initialize the distributed library and bind the worker to GPU.

        Return:
            True if distributed library is initialized successfully.
        """
        if self._args.distributed_impl:
            logger.info(
                'Distributed training is enabled - model: {}, distributed implementation: {}.'.format(
                    self._name, self._args.distributed_impl
                )
            )
            if self._args.distributed_impl == DistributedImpl.HOROVOD:
                import horovod.torch as hvd
                hvd.init()
                self._world_size = int(hvd.size())
                self._local_rank = int(hvd.local_rank())
                self._global_rank = int(hvd.rank())
            elif self._args.distributed_impl == DistributedImpl.DDP:
                if os.environ.get('WORLD_SIZE') is None or os.environ.get('LOCAL_RANK') is None:
                    logger.error(
                        'Can not find WORLD_SIZE or LOCAL_RANK in env variables - model: {},'
                        ' distributed implementation: {}.'.format(self._name, self._args.distributed_impl)
                    )
                    return False
                # torch >= 1.9.0a0 torch.distributed.elastic is used by default
                port = int(os.environ.get('MASTER_PORT', '29500')) + 1
                os.environ['MASTER_PORT'] = str(port)
                addr = os.environ['MASTER_ADDR']
                self._global_rank = int(os.environ['RANK'])
                self._local_rank = int(os.environ['LOCAL_RANK'])
                self._world_size = int(os.environ['WORLD_SIZE'])
                logger.debug('ip:{},port:{},rank:{},world:{}'.format(addr, port, self._global_rank, self._world_size))
                store = PrefixStore(
                    self._name, TCPStore(addr, port, self._world_size, self._global_rank == 0, timedelta(seconds=300))
                )
                torch.distributed.init_process_group(
                    backend=self._args.distributed_backend.value,
                    timeout=timedelta(seconds=300),
                    rank=self._global_rank,
                    world_size=self._world_size,
                    store=store
                )

            else:
                logger.error(
                    'Unsupported distributed implementation - model: {}, distributed implementation: {}.'.format(
                        self._name, self._args.distributed_impl
                    )
                )
                return False

            if self._gpu_available:
                torch.cuda.set_device(self._local_rank)

        return True

    def _init_dataloader(self):
        """Initialize the dataloader.

        Return:
            True if dataloader is created successfully.
        """
        train_sampler = None
        if self._args.distributed_impl:
            if self._args.distributed_impl == DistributedImpl.HOROVOD:
                import horovod.torch as hvd

                train_sampler = \
                    torch.utils.data.distributed.DistributedSampler(
                        self._dataset,
                        num_replicas=hvd.size(),
                        rank=hvd.rank()
                    )
            elif self._args.distributed_impl == DistributedImpl.DDP:
                try:
                    train_sampler = \
                        torch.utils.data.distributed.DistributedSampler(
                            self._dataset
                        )
                except BaseException as e:
                    logger.error(
                        'Init dataloader failed - model: {}, distributed implementation: {}, message: {}.'.format(
                            self._name, self._args.distributed_impl, str(e)
                        )
                    )
                    return False
            else:
                logger.error(
                    'Unsupported distributed implementation - model: {}, distributed implementation: {}.'.format(
                        self._name, self._args.distributed_impl
                    )
                )
                return False

        self._dataloader = DataLoader(
            dataset=self._dataset,
            batch_size=self._args.batch_size,
            shuffle=False,
            num_workers=self._args.num_workers,
            sampler=train_sampler,
            drop_last=True,
            pin_memory=self._args.pin_memory
        )

        return True

    def _create_optimizer(self):
        """Create the optimzier instance used for training and wrap with distributed library if need.

        Return:
            True if optimizer instance is created successfully.
        """
        if self._args.distributed_impl == DistributedImpl.DDP:
            self._model = torch.nn.parallel.DistributedDataParallel(
                self._model, device_ids=[self._local_rank], output_device=self._local_rank
            )

        if self._optimizer_type == Optimizer.SGD:
            self._optimizer = torch.optim.SGD(
                self._model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-4, nesterov=True
            )
        elif self._optimizer_type == Optimizer.ADAM:
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08)
        elif self._optimizer_type == Optimizer.ADAMW:
            if hasattr(torch.optim, 'AdamW'):
                self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08)
            else:
                self._optimizer = transformers.AdamW(self._model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08)
        else:
            self._optimizer = None

        if not self._optimizer:
            logger.error(
                'Create optimizer failed - model: {}, optimizer type: {}.'.format(self._name, self._optimizer_type)
            )
            return False

        if self._args.distributed_impl == DistributedImpl.HOROVOD:
            import horovod.torch as hvd
            self._optimizer = hvd.DistributedOptimizer(
                self._optimizer,
                named_parameters=self._model.named_parameters(),
                compression=hvd.Compression.none,
                op=hvd.Average
            )
            hvd.broadcast_parameters(self._model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(self._optimizer, root_rank=0)

        return True

    def _is_finished(self, curr_step, curr_time, check_frequency=100):
        """Judge whether the benchmarking should be stopped early or not.

        Args:
            curr_step (int): the current benchmarking step.
            curr_time (float): the current time in seconds got from time.time().
            check_frequency (int): the frequency (step numbers) to check if benchmark should be stopped.

        Return:
            True if the benchmarking should be stopped.
        """
        is_finished = int(super()._is_finished(curr_step, curr_time))
        if self._args.duration > 0:
            if curr_step % check_frequency == 0:
                # sync is_finished in distributed mode
                # if any rank is_finished is True, all ranks should be finished
                if self._args.distributed_impl == DistributedImpl.DDP:
                    tensor = torch.IntTensor([is_finished])
                    if self._args.distributed_backend == DistributedBackend.NCCL:
                        tensor = tensor.cuda()
                    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
                    is_finished = tensor.tolist()[0]
            else:
                is_finished = 0

        return (is_finished == 1)

    def _sync_result(self, result):
        """Function to reduce the result to rank 0.

        Args:
            result (list): The result data to sync.

        Return:
            Result if reduce result data successfully, otherwise None.
        """
        result = super()._sync_result(result)
        if not result:
            return None

        try:
            if self._args.distributed_impl == DistributedImpl.DDP:
                if self._args.distributed_backend == DistributedBackend.NCCL:
                    tensor = torch.as_tensor(result).cuda()
                else:
                    tensor = torch.as_tensor(result)
                torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
                result = tensor.tolist()
        except BaseException as e:
            logger.error(
                'Sync train result failed - model: {}, distributed implementation: {}, message: {}.'.format(
                    self._name, self._args.distributed_impl, str(e)
                )
            )
            return None

        return result

    def _postprocess(self):
        """Postprocess/cleanup operations after the benchmarking.

        Return:
            True if _postprocess() succeed.
        """
        if not super()._postprocess():
            return False

        try:
            if self._args.distributed_impl == DistributedImpl.DDP:
                torch.distributed.barrier()
                torch.distributed.destroy_process_group()
        except BaseException as e:
            self._result.set_return_code(ReturnCode.DISTRIBUTED_SETTING_DESTROY_FAILURE)
            logger.error(
                'Post process failed - model: {}, distributed implementation: {}, message: {}.'.format(
                    self._name, self._args.distributed_impl, str(e)
                )
            )
            return False

        if self._gpu_available:
            torch.cuda.synchronize()
        del self._target
        del self._optimizer
        del self._model
        if self._gpu_available:
            torch.cuda.empty_cache()

        return True

    def _cal_params_count(self):
        """Calculate the parameters scale of the model.

        Return:
            The count of trainable parameters.
        """
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    def _timer(self):
        """Returns the current time which ensures all previous CUDA events have been finished.

        If there is no GPU present, this defaults to `time.time()`; otherwise it will
        synchronize CUDA before measuring the time.

        Returns:
            Current time in second.
        """
        if self._gpu_available:
            torch.cuda.synchronize()
        return time.time()

    def _benchmark(self):
        """Wrap super._benchmark with profiler context if enabled by environment variable.

        Run the benchmark then handle post-run model log save/compare.
        Set SB_ENABLE_PYTORCH_PROFILER='1' to enable profiling.
        """
        # Check if this is a Nvidia GPU
        if not (torch.cuda.is_available() and torch.version.cuda is not None):
            ok = super()._benchmark()
            self._post_run_model_log()
            return ok

        # Check if profiling is enabled via environment variable
        enable_profiler = os.environ.get('SB_ENABLE_PYTORCH_PROFILER', '0') == '1'

        if not enable_profiler:
            # Run without profiling
            ok = super()._benchmark()
            self._post_run_model_log()
            return ok

        # Run with profiling enabled
        logger.info('PyTorch profiler enabled for model: {}'.format(self._name))
        ret = None

        from torch.profiler import profile, ProfilerActivity
        from torch.autograd import DeviceType
        import json

        if self._local_rank is None:
            local_rank = 0
        else:
            local_rank = self._local_rank

        diag_agent_prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True)
        dump_file_dir = os.environ.get('SB_TORCH_PROFILER_TRACE_DIR', '.')
        diag_agent_dump_file_path = f'{dump_file_dir}/torch-profiler-sb-{self._name}-{local_rank}.json'
        diag_agent_prof.__enter__()

        ret = super()._benchmark()

        diag_agent_prof.__exit__(None, None, None)
        diag_agent_events = []
        for event in diag_agent_prof.events():
            if event.device_type != DeviceType.CPU:
                continue
            diag_agent_event = {
                'name': event.name,
                'input_shapes': event.input_shapes,
                'input_values': event.concrete_inputs,
            }
            diag_agent_event['cpu_time'] = event.cpu_time
            diag_agent_event['gpu_time'] = event.cuda_time
            diag_agent_event['start_time'] = event.time_range.start
            diag_agent_events.append(diag_agent_event)
        with open(diag_agent_dump_file_path, 'w') as f:
            json.dump(diag_agent_events, f, sort_keys=True)

        # Handle post-run model log save/compare regardless of profiling
        self._post_run_model_log()
        return ret
