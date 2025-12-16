# HuggingFace Integration - Quick Reference

## 🚀 Quick Start

### 1. Benchmark a HuggingFace Model

```yaml
# config.yaml
superbench:
  enable:
    - tensorrt-inference
  var:
    pytorch_models:
      - model_identifier: 'Qwen/Qwen2.5-7B-Instruct'
        model_source: 'huggingface'
        batch_size: 8
        precision: 'fp16'
```

```bash
sb run --config config.yaml
```

### 2. Authenticate for Gated Models

```bash
export HF_TOKEN=your_token_here
sb run --config config.yaml
```

### 3. Compare In-house vs HuggingFace

```yaml
pytorch_models:
  - model_identifier: 'bert-large'
    model_source: 'in-house'
  - model_identifier: 'bert-large-uncased'
    model_source: 'huggingface'
```

## 📚 Documentation

- **[User Guide](docs/user-tutorial/huggingface-models.md)** - Complete tutorial
- **[Design Document](docs/design-docs/huggingface-model-integration.md)** - Technical details
- **[Example Config](examples/hf_models_example.yaml)** - Real-world examples
- **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)** - What's been built

## 🎯 Key Features

✅ Load any HuggingFace model  
✅ Compare in-house vs HF models  
✅ Support for latest models (Qwen, DeepSeek)  
✅ Authentication for gated models  
✅ Automatic model caching  
✅ ONNX export support  

## 📦 New Files

| File | Purpose |
|------|---------|
| `model_source_config.py` | Configuration dataclass |
| `huggingface_model_loader.py` | HF model loader |
| `_export_torch_to_onnx.py` | Enhanced (HF support added) |
| `test_model_source_config.py` | Unit tests |
| `test_huggingface_loader.py` | Unit tests |

## 🔧 Python API

```python
from superbench.benchmarks.micro_benchmarks import (
    HuggingFaceModelLoader,
    ModelSourceConfig,
    torch2onnxExporter
)

# Load model
loader = HuggingFaceModelLoader()
model, config, tokenizer = loader.load_model('Qwen/Qwen2.5-7B')

# Export to ONNX
exporter = torch2onnxExporter()
onnx_path = exporter.export_huggingface_model(
    'Qwen/Qwen2.5-7B',
    batch_size=8
)
```

## ✅ Supported Models

**Well-Tested**: BERT, GPT-2, Llama, Qwen, Mixtral, BLOOM, OPT  
**Experimental**: DeepSeek, Falcon, MPT, Mistral

[Full list in user guide](docs/user-tutorial/huggingface-models.md#supported-model-architectures)

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Check model ID on huggingface.co |
| Auth required | Set `HF_TOKEN` environment variable |
| Out of memory | Reduce batch_size or use fp16 |
| ONNX export failed | Check architecture compatibility |

[Full troubleshooting guide](docs/user-tutorial/huggingface-models.md#troubleshooting)

## 📊 Status

- **Phase 1**: ✅ Complete (Core implementation)
- **Phase 2**: 🔄 Next (Benchmark integration)
- **Phase 3**: 📅 Future (Comparison framework)

Last Updated: December 16, 2025
