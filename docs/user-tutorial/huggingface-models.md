# HuggingFace Model Integration - User Guide

## Overview

SuperBench now supports loading models directly from Hugging Face Hub for benchmarking. This allows you to:

- Benchmark any public model from Hugging Face
- Compare in-house model definitions with official HF implementations
- Test latest models (Qwen, DeepSeek, etc.) without code changes
- Use pre-trained weights for more realistic performance testing

## Quick Start

### 1. Basic HuggingFace Model

```yaml
superbench:
  enable:
    - tensorrt-inference
  var:
    pytorch_models:
      - model_identifier: 'bert-base-uncased'
        model_source: 'huggingface'
        batch_size: 32
        seq_length: 512
        precision: 'fp16'
```

### 2. Gated Model (Requires Authentication)

```yaml
pytorch_models:
  - model_identifier: 'meta-llama/Llama-2-7b-hf'
    model_source: 'huggingface'
    batch_size: 8
    hf_token: '${HF_TOKEN}'  # Set via environment variable
```

Set your token:
```bash
export HF_TOKEN=your_huggingface_token
sb run --config config.yaml
```

### 3. Compare In-house vs HuggingFace

```yaml
pytorch_models:
  # In-house definition
  - model_identifier: 'bert-large'
    model_source: 'in-house'
    batch_size: 32
  
  # Official HF model
  - model_identifier: 'bert-large-uncased'
    model_source: 'huggingface'
    batch_size: 32
```

## Configuration Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `model_identifier` | Yes | - | Model name (in-house) or HF model ID |
| `model_source` | Yes | `in-house` | Source: `in-house` or `huggingface` |
| `batch_size` | No | 32 | Batch size for inference |
| `seq_length` | No | 512 | Sequence length |
| `precision` | No | `fp32` | Precision: `fp32`, `fp16`, `int8` |
| `torch_dtype` | No | `float32` | Model dtype: `float32`, `float16`, `bfloat16` |
| `hf_token` | No | `None` | HuggingFace auth token |
| `trust_remote_code` | No | `false` | Allow remote code execution |
| `revision` | No | `main` | Model version/commit/tag |
| `cache_dir` | No | `~/.cache/huggingface` | Model cache directory |

## Supported Model Architectures

### Well-Tested (Recommended)
- **BERT Family**: BERT, RoBERTa, DistilBERT, ALBERT, ELECTRA
- **GPT Family**: GPT-2, GPT-Neo, GPT-J
- **Llama Family**: Llama, Llama 2, Llama 3
- **Qwen Family**: Qwen, Qwen2
- **Others**: Mixtral, BLOOM, OPT, BART, T5

### Experimental
- DeepSeek, Falcon, MPT, StableLM, Phi, Mistral

*Note: Experimental models may require special handling for ONNX export.*

## Examples

### Example 1: Benchmark Qwen 2.5

```yaml
superbench:
  enable:
    - tensorrt-inference
  var:
    pytorch_models:
      - model_identifier: 'Qwen/Qwen2.5-7B-Instruct'
        model_source: 'huggingface'
        batch_size: 8
        seq_length: 4096
        precision: 'fp16'
        torch_dtype: 'float16'
```

### Example 2: Benchmark DeepSeek Coder

```yaml
pytorch_models:
  - model_identifier: 'deepseek-ai/deepseek-coder-6.7b-instruct'
    model_source: 'huggingface'
    batch_size: 8
    seq_length: 2048
    precision: 'fp16'
```

### Example 3: Multiple Model Sizes

```yaml
pytorch_models:
  - model_identifier: 'Qwen/Qwen2.5-0.5B'
    model_source: 'huggingface'
    batch_size: 64
  
  - model_identifier: 'Qwen/Qwen2.5-1.5B'
    model_source: 'huggingface'
    batch_size: 32
  
  - model_identifier: 'Qwen/Qwen2.5-7B'
    model_source: 'huggingface'
    batch_size: 8
```

## Python API

### Loading a Model

```python
from superbench.benchmarks.micro_benchmarks import (
    HuggingFaceModelLoader,
    ModelSourceConfig
)

# Method 1: Direct loading
loader = HuggingFaceModelLoader()
model, config, tokenizer = loader.load_model('Qwen/Qwen2.5-7B')

# Method 2: Using config
config = ModelSourceConfig(
    source='huggingface',
    identifier='Qwen/Qwen2.5-7B',
    torch_dtype='float16'
)
model, cfg, tokenizer = loader.load_model_from_config(config)
```

### Exporting to ONNX

```python
from superbench.benchmarks.micro_benchmarks import torch2onnxExporter

exporter = torch2onnxExporter()
onnx_path = exporter.export_huggingface_model(
    model_identifier='Qwen/Qwen2.5-7B',
    batch_size=8,
    seq_length=2048,
    torch_dtype='float16'
)
```

### Checking Model Compatibility

```python
loader = HuggingFaceModelLoader()

# Get model info without downloading
info = loader.get_model_info('Qwen/Qwen2.5-7B')
print(f"Architecture: {info['architecture']}")
print(f"ONNX Compatible: {info['onnx_compatible']}")

# List supported architectures
architectures = loader.list_supported_architectures()
print(f"Supported: {architectures['supported']}")
print(f"Experimental: {architectures['experimental']}")
```

## Troubleshooting

### Issue: Model Not Found

```
Error: Model 'xyz/model' not found on Hugging Face Hub
```

**Solution**: Verify the model ID at https://huggingface.co/models

### Issue: Authentication Required

```
Error: Repository is gated. You must be authenticated.
```

**Solution**: 
1. Get your token from https://huggingface.co/settings/tokens
2. Set the environment variable: `export HF_TOKEN=your_token`
3. Or provide in config: `hf_token: '${HF_TOKEN}'`

### Issue: Out of Memory

```
Error: CUDA out of memory
```

**Solution**:
- Reduce `batch_size`
- Use `device_map: 'auto'` for model sharding
- Use lower precision (`torch_dtype: 'float16'`)

### Issue: ONNX Export Failed

```
Error: Failed to export model to ONNX
```

**Solution**:
- Check if model architecture is supported
- Try `torch_dtype='float32'` instead of float16
- Check compatibility: `loader.validate_model_compatibility(model, config)`

### Issue: Trust Remote Code Warning

```
Warning: trust_remote_code=True allows execution of remote code
```

**Solution**: Only set `trust_remote_code: true` for trusted models. Review the model's code on HuggingFace before enabling.

## Performance Tips

1. **Use FP16**: Set `torch_dtype: 'float16'` for better performance
2. **Cache Models**: Models are cached after first download
3. **Batch Size**: Start with smaller batch sizes for large models
4. **Sequence Length**: Adjust based on your use case
5. **Device Mapping**: For very large models, use `device_map: 'auto'`

## Security Considerations

- **Tokens**: Never commit tokens to version control. Use environment variables.
- **Remote Code**: Set `trust_remote_code: false` unless you trust the model source.
- **Model Validation**: Always check model compatibility before running.

## Migration from In-house Models

Existing configurations continue to work without changes:

```yaml
# Before (still works)
pytorch_models:
  - bert-large
  - gpt2-large

# After (enhanced, optional)
pytorch_models:
  - model_identifier: 'bert-large'
    model_source: 'in-house'  # Optional, this is default
  - model_identifier: 'gpt2-large'
    model_source: 'in-house'
```

## FAQ

**Q: Can I use fine-tuned models?**  
A: Yes! Any model on Hugging Face Hub can be used, including fine-tuned versions.

**Q: Do I need to download models every time?**  
A: No, models are cached locally after first download.

**Q: Can I use models from private repos?**  
A: Yes, provide your HF token with `hf_token` parameter.

**Q: What if my model architecture isn't listed as supported?**  
A: You can still try it! The loader will warn you if there might be issues.

**Q: Can I compare multiple versions of the same model?**  
A: Yes, use the `revision` parameter to specify different versions.

## Additional Resources

- [Design Document](../design-docs/huggingface-model-integration.md)
- [HuggingFace Hub](https://huggingface.co/models)
- [ONNX Export Guide](https://pytorch.org/docs/stable/onnx.html)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [Design Document](../design-docs/huggingface-model-integration.md)
3. Open an issue on GitHub with:
   - Model identifier
   - Configuration used
   - Error message and logs
