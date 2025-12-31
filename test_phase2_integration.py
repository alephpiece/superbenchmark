#!/usr/bin/env python3
"""
Phase 2 Integration Test for HuggingFace Model Loading

Tests the complete workflow:
1. YAML config parsing
2. ModelSourceConfig creation
3. HuggingFaceModelLoader usage
4. Model instantiation in benchmark
"""

import sys
import os
import tempfile
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from superbench.benchmarks.micro_benchmarks.model_source_config import ModelSourceConfig
from superbench.benchmarks.micro_benchmarks.huggingface_model_loader import HuggingFaceModelLoader


def test_yaml_config_parsing():
    """Test parsing YAML config with HuggingFace parameters."""
    print("=" * 80)
    print("TEST 1: YAML Configuration Parsing")
    print("=" * 80)
    
    yaml_content = """
superbench:
  benchmarks:
    pytorch-bert-hf:
      models:
        - bert-base
      parameters:
        model_source: huggingface
        model_identifier: bert-base-uncased
        num_classes: 2
        seq_len: 128
    """
    
    try:
        config = yaml.safe_load(yaml_content)
        bench_config = config['superbench']['benchmarks']['pytorch-bert-hf']['parameters']
        
        print(f"✓ YAML parsed successfully")
        print(f"  - model_source: {bench_config.get('model_source')}")
        print(f"  - model_identifier: {bench_config.get('model_identifier')}")
        print(f"  - num_classes: {bench_config.get('num_classes')}")
        
        assert bench_config['model_source'] == 'huggingface'
        assert bench_config['model_identifier'] == 'bert-base-uncased'
        print("✓ All assertions passed\n")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {str(e)}\n")
        return False


def test_model_source_config_from_yaml():
    """Test creating ModelSourceConfig from YAML parameters."""
    print("=" * 80)
    print("TEST 2: ModelSourceConfig from YAML Parameters")
    print("=" * 80)
    
    yaml_params = {
        'model_source': 'huggingface',
        'model_identifier': 'bert-base-uncased',
        'hf_token': None,
        'trust_remote_code': False
    }
    
    try:
        config = ModelSourceConfig(
            source=yaml_params['model_source'],
            identifier=yaml_params['model_identifier'],
            hf_token=yaml_params.get('hf_token'),
            trust_remote_code=yaml_params.get('trust_remote_code', False)
        )
        
        print(f"✓ ModelSourceConfig created successfully")
        print(f"  - is_huggingface: {config.is_huggingface()}")
        print(f"  - identifier: {config.identifier}")
        print(f"  - trust_remote_code: {config.trust_remote_code}")
        
        assert config.is_huggingface()
        assert config.identifier == 'bert-base-uncased'
        print("✓ All assertions passed\n")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {str(e)}\n")
        return False


def test_huggingface_model_loading():
    """Test loading a small model from HuggingFace."""
    print("=" * 80)
    print("TEST 3: HuggingFace Model Loading")
    print("=" * 80)
    
    try:
        # Use a very small model for quick testing
        config = ModelSourceConfig(
            source='huggingface',
            identifier='prajjwal1/bert-tiny',  # Tiny BERT for fast testing
            torch_dtype='float32'
        )
        
        print(f"Loading model: {config.identifier}")
        loader = HuggingFaceModelLoader()
        
        model, model_config, tokenizer = loader.load_model_from_config(config)
        
        print(f"✓ Model loaded successfully")
        print(f"  - Model type: {type(model).__name__}")
        print(f"  - Config type: {type(model_config).__name__}")
        print(f"  - Hidden size: {model_config.hidden_size}")
        print(f"  - Num layers: {model_config.num_hidden_layers}")
        print(f"  - Num attention heads: {model_config.num_attention_heads}")
        
        if tokenizer:
            print(f"  - Tokenizer vocab size: {len(tokenizer)}")
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  - Total parameters: {num_params:,} ({num_params/1e6:.2f}M)")
        
        assert model is not None
        assert model_config is not None
        print("✓ All assertions passed\n")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def test_complete_workflow():
    """Test the complete workflow from YAML to model."""
    print("=" * 80)
    print("TEST 4: Complete Workflow (YAML → Config → Model)")
    print("=" * 80)
    
    yaml_content = """
model_source: huggingface
model_identifier: prajjwal1/bert-tiny
hf_token: null
trust_remote_code: false
torch_dtype: float32
    """
    
    try:
        # Parse YAML
        params = yaml.safe_load(yaml_content)
        print(f"✓ YAML parsed")
        
        # Create ModelSourceConfig
        config = ModelSourceConfig(
            source=params['model_source'],
            identifier=params['model_identifier'],
            hf_token=params.get('hf_token'),
            trust_remote_code=params.get('trust_remote_code', False),
            torch_dtype=params.get('torch_dtype', 'float32')
        )
        print(f"✓ ModelSourceConfig created")
        
        # Load model
        loader = HuggingFaceModelLoader()
        model, model_config, tokenizer = loader.load_model_from_config(config)
        print(f"✓ Model loaded from HuggingFace")
        
        # Verify model is callable
        import torch
        batch_size = 2
        seq_length = 16
        
        # Create sample input
        input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        print(f"  - Testing model forward pass...")
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        print(f"  - Output shape: {outputs.last_hidden_state.shape}")
        expected_shape = (batch_size, seq_length, model_config.hidden_size)
        assert outputs.last_hidden_state.shape == expected_shape
        
        print(f"✓ Model forward pass successful")
        print("✓ All assertions passed\n")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 2 integration tests."""
    print("\n" + "=" * 80)
    print("PHASE 2 INTEGRATION TESTS")
    print("Testing HuggingFace Integration in SuperBench Workflow")
    print("=" * 80 + "\n")
    
    tests = [
        ("YAML Config Parsing", test_yaml_config_parsing),
        ("ModelSourceConfig from YAML", test_model_source_config_from_yaml),
        ("HuggingFace Model Loading", test_huggingface_model_loading),
        ("Complete Workflow", test_complete_workflow),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test '{test_name}' crashed: {str(e)}\n")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 2 integration tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
