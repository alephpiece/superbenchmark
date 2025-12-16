# Hugging Face Model Integration - Design Document

**Author**: SuperBench Team  
**Date**: December 15, 2025  
**Status**: Design Phase  
**Version**: 1.0

## Executive Summary

This document describes the design for integrating Hugging Face model loading capabilities into SuperBench, enabling users to benchmark any pre-trained model from Hugging Face Hub alongside existing in-house model definitions. The integration maintains backward compatibility while providing flexibility to benchmark the latest models (Qwen, DeepSeek, etc.) through ONNX and TensorRT workflows.

## Table of Contents

1. [Background](#background)
2. [Goals and Non-Goals](#goals-and-non-goals)
3. [Architecture Overview](#architecture-overview)
4. [Detailed Design](#detailed-design)
5. [API Design](#api-design)
6. [Configuration Schema](#configuration-schema)
7. [Implementation Plan](#implementation-plan)
8. [Testing Strategy](#testing-strategy)
9. [Migration and Compatibility](#migration-and-compatibility)
10. [Future Enhancements](#future-enhancements)

---

## Background

### Current State

SuperBench currently supports model inference benchmarking through:
- **In-house model definitions**: Hardcoded model configurations (BERT, GPT2, Llama, Mixtral, LSTM)
- **TorchVision models**: Standard vision models from torchvision library
- **Workflow**: Model Config → Synthetic Data → ONNX Export → TensorRT Conversion → Inference Benchmarking

### Problem Statement

1. **Limited Model Coverage**: Only predefined models can be benchmarked
2. **No Pre-trained Weights**: Models use synthetic configs without actual pre-trained weights
3. **Manual Addition Required**: Adding new models requires code changes
4. **Cannot Test Latest Models**: Unable to benchmark recent models like Qwen, DeepSeek without code updates
5. **No Source Comparison**: Cannot compare in-house configs vs. official HF implementations

### Motivation

- Enable benchmarking of any model available on Hugging Face Hub
- Support loading pre-trained weights for more realistic performance testing
- Allow users to specify models via configuration without code changes
- Compare performance between in-house definitions and official HF models
- Stay current with latest model releases (Qwen 2.5, DeepSeek V3, etc.)

---

## Goals and Non-Goals

### Goals

✅ **G1**: Support loading models from Hugging Face Hub with pre-trained weights  
✅ **G2**: Maintain backward compatibility with existing in-house model definitions  
✅ **G3**: Allow user-defined model selection via configuration  
✅ **G4**: Enable side-by-side comparison of in-house vs. HF models  
✅ **G5**: Support ONNX export and TensorRT conversion for HF models  
✅ **G6**: Provide clear error messages for incompatible models  
✅ **G7**: Cache downloaded models to avoid repeated downloads  
✅ **G8**: Support authenticated access to private/gated models  

### Non-Goals

❌ **NG1**: Integration with TensorRT-LLM API (out of scope for this phase)  
❌ **NG2**: Fine-tuning or training models (focus on inference only)  
❌ **NG3**: Automatic model optimization (user responsible for optimizations)  
❌ **NG4**: Supporting non-PyTorch models (only PyTorch-based models from HF)  
❌ **NG5**: Real-time model updates from HF Hub during benchmark execution  

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      SuperBench CLI/Config                       │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
        ┌───────▼────────┐           ┌─────────▼──────────┐
        │   In-House     │           │   Hugging Face     │
        │  Model Loader  │           │   Model Loader     │
        │   (Existing)   │           │      (New)         │
        └───────┬────────┘           └─────────┬──────────┘
                │                               │
                │    ┌──────────────────────────┘
                │    │
        ┌───────▼────▼────────┐
        │  Enhanced ONNX      │
        │     Exporter        │
        │  (torch2onnx+)      │
        └───────┬─────────────┘
                │
        ┌───────▼─────────────┐
        │   ONNX Model File   │
        └───────┬─────────────┘
                │
        ┌───────▼─────────────┐
        │  TensorRT Engine    │
        │     Builder         │
        └───────┬─────────────┘
                │
        ┌───────▼─────────────┐
        │  Inference Bench    │
        │  (TRT/ORT/Torch)    │
        └───────┬─────────────┘
                │
        ┌───────▼─────────────┐
        │  Results & Metrics  │
        │   + Comparison      │
        └─────────────────────┘
```

### Component Interaction Flow

```
User Config (YAML)
    │
    ├─► Parse model_source: 'huggingface' or 'in-house'
    │
    ├─► HuggingFaceModelLoader
    │   ├─► Authenticate (if token provided)
    │   ├─► Download/Load model from HF Hub
    │   ├─► Validate model architecture compatibility
    │   └─► Return PyTorch model + config
    │
    ├─► Enhanced torch2onnxExporter
    │   ├─► Receive model from loader
    │   ├─► Generate appropriate dummy inputs
    │   ├─► Export to ONNX format
    │   └─► Validate ONNX model
    │
    ├─► TensorRT/ORT Benchmark
    │   ├─► Load ONNX model
    │   ├─► Run inference benchmarks
    │   └─► Collect metrics
    │
    └─► Results Comparison (optional)
        ├─► Compare in-house vs HF versions
        ├─► Generate comparison report
        └─► Output metrics
```

---

## Detailed Design

### 1. HuggingFaceModelLoader Class

**Location**: `superbench/benchmarks/micro_benchmarks/huggingface_model_loader.py`

```python
class HuggingFaceModelLoader:
    """Loads models from Hugging Face Hub for benchmarking."""
    
    def __init__(self, cache_dir=None, token=None):
        """
        Args:
            cache_dir: Directory to cache downloaded models
            token: HuggingFace authentication token for private models
        """
        
    def load_model(self, model_identifier, model_type='auto', 
                   torch_dtype=None, device='cuda'):
        """
        Load a model from Hugging Face Hub.
        
        Args:
            model_identifier: HF model ID (e.g., 'Qwen/Qwen2.5-7B')
            model_type: Model architecture type for targeted loading
            torch_dtype: Data type for model weights
            device: Device to load model on
            
        Returns:
            Tuple of (model, config, tokenizer)
            
        Raises:
            ModelNotFoundError: If model doesn't exist
            ModelIncompatibleError: If model cannot be exported to ONNX
        """
        
    def validate_model_compatibility(self, model):
        """
        Check if model can be exported to ONNX.
        
        Returns:
            Tuple of (is_compatible, reason)
        """
        
    def get_model_info(self, model_identifier):
        """
        Retrieve model metadata without downloading.
        
        Returns:
            Dict with model info (size, architecture, etc.)
        """
        
    def list_supported_architectures(self):
        """
        Return list of model architectures known to work with ONNX export.
        """
```

**Key Features**:
- Automatic model architecture detection using `AutoModel`
- Caching to avoid repeated downloads
- Memory-efficient loading with device mapping
- Authentication support for private/gated models
- Pre-flight validation before ONNX export
- Detailed error messages with suggestions

**Supported Model Types**:
- `BertModel`, `RobertaModel` (BERT family)
- `GPT2Model`, `GPTNeoModel`, `GPTJModel` (GPT family)
- `LlamaModel` (Llama, Llama 2, Llama 3)
- `MixtralModel` (Mixtral models)
- `QwenModel`, `Qwen2Model` (Qwen family)
- Any model with compatible architecture

### 2. Enhanced torch2onnxExporter Class

**Location**: `superbench/benchmarks/micro_benchmarks/_export_torch_to_onnx.py`

**New Methods**:

```python
class torch2onnxExporter:
    """Enhanced PyTorch model to ONNX exporter with HF support."""
    
    def __init__(self):
        # Existing initialization
        self.hf_loader = HuggingFaceModelLoader()
        
    def export_huggingface_model(self, model_identifier, batch_size=1, 
                                  seq_length=512, torch_dtype='float32'):
        """
        Export a Hugging Face model to ONNX format.
        
        Args:
            model_identifier: HF model ID (e.g., 'meta-llama/Llama-2-7b')
            batch_size: Batch size for input
            seq_length: Sequence length for input
            torch_dtype: Data type for model
            
        Returns:
            str: Path to exported ONNX model
            
        Raises:
            ExportError: If ONNX export fails
        """
        
    def check_huggingface_model(self, model_identifier):
        """
        Check if HF model exists and is compatible.
        
        Returns:
            bool: True if model can be loaded and exported
        """
        
    def _generate_inputs_for_model(self, model, config, batch_size, seq_length):
        """
        Generate appropriate dummy inputs based on model architecture.
        
        Returns:
            Dict of input tensors
        """
        
    def _get_dynamic_axes(self, model_type):
        """
        Get dynamic axes configuration for specific model types.
        
        Returns:
            Dict of dynamic axes
        """
```

**Enhancements**:
- New `model_source` parameter in existing methods
- Support for both in-house and HF models in unified interface
- Automatic input shape detection based on model config
- Model-specific ONNX export settings
- Progress tracking for large model exports

### 3. ModelSourceConfig Class

**Location**: `superbench/benchmarks/micro_benchmarks/model_source_config.py`

```python
@dataclass
class ModelSourceConfig:
    """Configuration for model source and loading."""
    
    source: str  # 'in-house' or 'huggingface'
    identifier: str  # Model name/ID
    hf_token: Optional[str] = None
    torch_dtype: str = 'float32'
    trust_remote_code: bool = False
    revision: Optional[str] = None  # Specific commit/tag
    
    def validate(self):
        """Validate configuration parameters."""
        
    def to_dict(self):
        """Convert to dictionary for serialization."""
```

### 4. Benchmark Integration

**Update to `tensorrt_inference_performance.py`**:

```python
class TensorRTInferenceBenchmark(MicroBenchmarkWithInvoke):
    
    def add_parser_arguments(self):
        # Existing arguments...
        
        self._parser.add_argument(
            '--model_source',
            type=str,
            choices=['in-house', 'huggingface'],
            default='in-house',
            help='Source of model: in-house definitions or Hugging Face Hub'
        )
        
        self._parser.add_argument(
            '--model_identifier',
            type=str,
            help='Model identifier (in-house name or HF model ID)'
        )
        
        self._parser.add_argument(
            '--hf_token',
            type=str,
            help='HuggingFace authentication token for private models'
        )
        
    def _preprocess(self):
        # Enhanced to handle both sources
        exporter = torch2onnxExporter()
        
        for model_spec in self._args.pytorch_models:
            if self._args.model_source == 'huggingface':
                onnx_model = exporter.export_huggingface_model(
                    model_spec,
                    self._args.batch_size,
                    self._args.seq_length
                )
            else:
                # Existing in-house logic
                onnx_model = self._export_inhouse_model(model_spec)
```

### 5. Comparison Framework

**Location**: `superbench/benchmarks/micro_benchmarks/model_comparison.py`

```python
class ModelComparison:
    """Compare benchmark results between different model sources."""
    
    def __init__(self):
        self.results = {}
        
    def add_result(self, model_name, source, metrics):
        """Add benchmark results for a model."""
        
    def compare(self, model_name):
        """
        Compare results for same model from different sources.
        
        Returns:
            ComparisonReport with detailed metrics
        """
        
    def generate_report(self, output_format='json'):
        """
        Generate comparison report.
        
        Supports: json, yaml, html, markdown
        """
```

---

## API Design

### Python API

```python
# Example 1: Load and benchmark HF model
from superbench.benchmarks.micro_benchmarks import (
    HuggingFaceModelLoader,
    torch2onnxExporter
)

loader = HuggingFaceModelLoader()
model, config, tokenizer = loader.load_model('Qwen/Qwen2.5-7B-Instruct')

exporter = torch2onnxExporter()
onnx_path = exporter.export_huggingface_model(
    'Qwen/Qwen2.5-7B-Instruct',
    batch_size=8,
    seq_length=512
)

# Example 2: Compare in-house vs HF
from superbench.benchmarks.micro_benchmarks import ModelComparison

comparison = ModelComparison()
comparison.add_result('bert-large', 'in-house', metrics1)
comparison.add_result('bert-large', 'huggingface', metrics2)
report = comparison.compare('bert-large')
```

### CLI API

```bash
# Benchmark HF model
sb run --config hf_models.yaml

# Compare models
sb run --config comparison.yaml --mode compare

# List supported HF models
sb list-models --source huggingface --architecture llama

# Validate model compatibility
sb validate-model Qwen/Qwen2.5-7B-Instruct
```

---

## Configuration Schema

### YAML Configuration

```yaml
# Example 1: Single HF model
superbench:
  version: v0.10
  superbench:
    enable:
      - tensorrt-inference
    var:
      models:
        - pytorch_models:
            - model_identifier: 'Qwen/Qwen2.5-7B-Instruct'
              model_source: 'huggingface'
              batch_size: 8
              seq_length: 512
              precision: 'fp16'
              hf_token: '${HF_TOKEN}'  # Optional

# Example 2: Compare in-house vs HF
superbench:
  version: v0.10
  superbench:
    enable:
      - tensorrt-inference
    var:
      models:
        - pytorch_models:
            - model_identifier: 'bert-large'
              model_source: 'in-house'
              batch_size: 32
            - model_identifier: 'bert-large-uncased'
              model_source: 'huggingface'
              batch_size: 32
      comparison:
        enabled: true
        models_to_compare:
          - base_name: 'bert-large'
            sources: ['in-house', 'huggingface']
        output_format: 'json'

# Example 3: Multiple HF models
superbench:
  version: v0.10
  superbench:
    enable:
      - tensorrt-inference
    var:
      hf_models:
        - identifier: 'Qwen/Qwen2.5-7B-Instruct'
          precision: 'fp16'
        - identifier: 'deepseek-ai/deepseek-coder-6.7b-instruct'
          precision: 'fp16'
        - identifier: 'meta-llama/Llama-2-7b-hf'
          precision: 'fp16'
          hf_token: '${HF_TOKEN}'  # Gated model
      benchmark_config:
        batch_sizes: [1, 8, 16, 32]
        seq_lengths: [128, 512, 1024]
        iterations: 1000
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_source` | string | 'in-house' | Source of model ('in-house' or 'huggingface') |
| `model_identifier` | string | required | Model name/ID |
| `hf_token` | string | null | HuggingFace auth token |
| `batch_size` | int | 32 | Batch size for inference |
| `seq_length` | int | 512 | Sequence length |
| `precision` | string | 'fp32' | Precision (fp32, fp16, int8) |
| `trust_remote_code` | bool | false | Allow remote code execution |
| `cache_dir` | string | null | Model cache directory |
| `revision` | string | 'main' | Model version/commit |
| `device_map` | string | 'auto' | Device mapping strategy |

---

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1-2)

**Tasks**:
1. Create `HuggingFaceModelLoader` class
   - Basic model loading from HF Hub
   - Caching mechanism
   - Authentication support
2. Create `ModelSourceConfig` dataclass
3. Add unit tests for loader
4. Documentation for new classes

**Deliverables**:
- `huggingface_model_loader.py` with full implementation
- `model_source_config.py` with configuration handling
- Unit tests with 80%+ coverage
- API documentation

### Phase 2: ONNX Export Enhancement (Week 3)

**Tasks**:
1. Extend `torch2onnxExporter` class
   - Add `export_huggingface_model()` method
   - Implement input generation logic
   - Add model validation
2. Implement compatibility checking
3. Add integration tests
4. Test with common HF models (BERT, GPT2, Llama)

**Deliverables**:
- Enhanced `_export_torch_to_onnx.py`
- Compatibility matrix documentation
- Integration tests
- Example exports for validation

### Phase 3: Benchmark Integration (Week 4)

**Tasks**:
1. Update `tensorrt_inference_performance.py`
   - Add new CLI arguments
   - Integrate HF loader
   - Handle both model sources
2. Update `ort_inference_performance.py`
3. Add error handling and logging
4. Test end-to-end workflow

**Deliverables**:
- Updated benchmark files
- End-to-end tests
- Error handling documentation
- CLI usage examples

### Phase 4: Comparison Framework (Week 5)

**Tasks**:
1. Create `ModelComparison` class
2. Implement comparison logic
3. Add report generation (JSON, YAML, HTML)
4. Create comparison examples

**Deliverables**:
- `model_comparison.py` implementation
- Report templates
- Comparison examples
- Usage documentation

### Phase 5: Testing & Documentation (Week 6)

**Tasks**:
1. Test with latest models (Qwen 2.5, DeepSeek V3)
2. Performance benchmarking
3. Create user guides
4. Update existing documentation
5. Add troubleshooting guide

**Deliverables**:
- Comprehensive test suite
- User guide document
- Troubleshooting guide
- Performance baseline data
- Migration guide

### Phase 6: Polish & Release (Week 7)

**Tasks**:
1. Code review and refactoring
2. Performance optimization
3. Final documentation review
4. Create release notes
5. Prepare examples and tutorials

**Deliverables**:
- Production-ready code
- Complete documentation
- Example configurations
- Release notes
- Tutorial notebooks

---

## Testing Strategy

### Unit Tests

```python
# test_huggingface_loader.py
def test_load_public_model():
    """Test loading a public model."""
    
def test_load_with_authentication():
    """Test loading gated model with token."""
    
def test_model_validation():
    """Test model compatibility checking."""
    
def test_cache_functionality():
    """Test model caching works correctly."""

# test_onnx_export.py
def test_export_huggingface_bert():
    """Test exporting BERT from HF."""
    
def test_export_with_different_dtypes():
    """Test export with various data types."""
    
def test_dynamic_axes_configuration():
    """Test dynamic axes are set correctly."""
```

### Integration Tests

```python
# test_end_to_end.py
def test_hf_to_tensorrt_pipeline():
    """Test complete HF → ONNX → TRT pipeline."""
    
def test_comparison_workflow():
    """Test comparing in-house vs HF models."""
    
def test_multiple_models_batch():
    """Test benchmarking multiple models."""
```

### Compatibility Testing

Test matrix:
- Models: BERT, GPT2, Llama 2, Qwen 2.5, DeepSeek
- Precisions: FP32, FP16, INT8
- Batch sizes: 1, 8, 32
- Sequence lengths: 128, 512, 1024

### Performance Testing

Benchmarks:
- Model loading time
- ONNX export time
- TensorRT conversion time
- Inference latency
- Memory usage

---

## Migration and Compatibility

### Backward Compatibility

✅ **Existing configs work without changes**
- Default `model_source='in-house'` maintains current behavior
- All existing model names continue to work
- No breaking changes to API

### Migration Guide

**For users with existing configs**:

```yaml
# Before (still works)
pytorch_models:
  - bert-large
  - gpt2-large

# After (enhanced)
pytorch_models:
  - model_identifier: 'bert-large'
    model_source: 'in-house'  # Optional, default
  - model_identifier: 'gpt2-large'
    model_source: 'in-house'
```

**Adding HF models**:

```yaml
# New HF models
pytorch_models:
  - model_identifier: 'Qwen/Qwen2.5-7B'
    model_source: 'huggingface'
    hf_token: '${HF_TOKEN}'  # If needed
```

### Deprecation Plan

No deprecations in this phase. Future consideration:
- Mark individual in-house configs as "legacy" when HF equivalents available
- Provide migration utility to convert configs
- Maintain backward compatibility for at least 2 major versions

---

## Error Handling

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `ModelNotFoundError` | Invalid HF model ID | Check model exists on HF Hub |
| `AuthenticationError` | Missing/invalid token | Provide valid HF token |
| `OOMError` | Model too large | Use smaller batch size or model sharding |
| `ONNXExportError` | Incompatible architecture | Check compatibility matrix |
| `ModelIncompatibleError` | Unsupported operations | Try different model variant |

### Error Message Examples

```
❌ Error: Cannot load model 'Invalid/Model-Name'
   Reason: Model not found on Hugging Face Hub
   Suggestion: Check the model ID at https://huggingface.co/models
   
❌ Error: ONNX export failed for 'model-name'
   Reason: Model contains unsupported operations
   Suggestion: Try exporting with torch_dtype='float32' or check compatibility
   
✅ Success: Model 'Qwen/Qwen2.5-7B' loaded successfully
   Cache location: /home/user/.cache/huggingface/hub
   Model size: 7.3 GB
```

---

## Security Considerations

### Authentication
- Support environment variables for tokens (`HF_TOKEN`)
- Never log tokens or include in outputs
- Support `.env` files for local development

### Trust Remote Code
- Default to `trust_remote_code=False`
- Warn users when enabling remote code execution
- Document security implications

### Model Validation
- Validate model architecture before loading
- Check model file signatures
- Scan for suspicious code patterns

---

## Performance Considerations

### Model Caching
- Use HuggingFace's built-in caching
- Configurable cache directory
- Automatic cache cleanup for old models

### Memory Management
- Lazy loading when possible
- Automatic device mapping for large models
- Clear GPU memory after ONNX export

### Parallel Processing
- Support batched model exports
- Parallel downloads for multiple models
- Async operations where applicable

---

## Future Enhancements

### Post-v1.0 Features

1. **Quantization Support**
   - Automatic quantization for HF models
   - Compare quantized vs full-precision

2. **Model Optimization**
   - Automatic graph optimization
   - Kernel fusion suggestions

3. **Fine-tuned Models**
   - Support custom fine-tuned models
   - LoRA adapter support

4. **Multi-modal Models**
   - Vision-language models (CLIP, LLaVA)
   - Audio models (Whisper)

5. **Model Serving Integration**
   - Export to NVIDIA Triton format
   - Generate serving configs

6. **Auto-tuning**
   - Automatic batch size selection
   - Optimal configuration search

7. **Model Zoo**
   - Curated list of tested models
   - Performance leaderboard

---

## Appendix

### A. Supported HF Model Architectures (Initial)

| Architecture | Status | Notes |
|--------------|--------|-------|
| BERT | ✅ Supported | Fully tested |
| GPT2 | ✅ Supported | Fully tested |
| Llama | ✅ Supported | Llama 2, 3 |
| Qwen | ⚠️ Beta | Testing in progress |
| DeepSeek | ⚠️ Beta | Testing in progress |
| Mixtral | ✅ Supported | MoE models |
| RoBERTa | ✅ Supported | BERT variant |
| GPT-Neo | ⚠️ Beta | Testing in progress |
| GPT-J | ⚠️ Beta | Testing in progress |
| Falcon | 🔄 Planned | Future support |

### B. Example Use Cases

**Use Case 1: Benchmark Latest Qwen Model**
```yaml
tensorrt_inference:
  pytorch_models:
    - model_identifier: 'Qwen/Qwen2.5-7B-Instruct'
      model_source: 'huggingface'
      precision: 'fp16'
```

**Use Case 2: Compare In-house BERT vs Official**
```yaml
tensorrt_inference:
  pytorch_models:
    - model_identifier: 'bert-large'
      model_source: 'in-house'
    - model_identifier: 'bert-large-uncased'
      model_source: 'huggingface'
  comparison:
    enabled: true
```

**Use Case 3: Test Multiple Model Sizes**
```yaml
tensorrt_inference:
  pytorch_models:
    - model_identifier: 'Qwen/Qwen2.5-0.5B'
      model_source: 'huggingface'
    - model_identifier: 'Qwen/Qwen2.5-1.5B'
      model_source: 'huggingface'
    - model_identifier: 'Qwen/Qwen2.5-7B'
      model_source: 'huggingface'
```

### C. References

- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub)
- [ONNX Export Guide](https://pytorch.org/docs/stable/onnx.html)
- [TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)
- [Transformers Library](https://huggingface.co/docs/transformers)

### D. Glossary

- **HF**: Hugging Face
- **ONNX**: Open Neural Network Exchange
- **TRT**: TensorRT
- **ORT**: ONNX Runtime
- **MoE**: Mixture of Experts
- **LoRA**: Low-Rank Adaptation

---

## Approval and Sign-off

| Role | Name | Date | Status |
|------|------|------|--------|
| Design Author | TBD | 2025-12-15 | ✅ Complete |
| Tech Lead Review | TBD | Pending | 🔄 In Progress |
| Architecture Review | TBD | Pending | ⏳ Scheduled |
| Security Review | TBD | Pending | ⏳ Scheduled |
| Final Approval | TBD | Pending | ⏳ Scheduled |

---

**Document History**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-15 | SuperBench Team | Initial design document |

