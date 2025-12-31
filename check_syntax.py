#!/usr/bin/env python3
"""Simple syntax check for new modules."""

import sys
import ast

def check_syntax(filepath):
    """Check if Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, "OK"
    except SyntaxError as e:
        return False, str(e)

files_to_check = [
    'superbench/benchmarks/micro_benchmarks/model_source_config.py',
    'superbench/benchmarks/micro_benchmarks/huggingface_model_loader.py',
    'superbench/benchmarks/micro_benchmarks/_export_torch_to_onnx.py',
]

print("=" * 70)
print("Syntax Check for HuggingFace Integration")
print("=" * 70 + "\n")

all_passed = True
for filepath in files_to_check:
    result, message = check_syntax(filepath)
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"{status}: {filepath}")
    if not result:
        print(f"   Error: {message}")
        all_passed = False

print("\n" + "=" * 70)
if all_passed:
    print("✅ All files have valid Python syntax!")
else:
    print("❌ Some files have syntax errors")
print("=" * 70)

sys.exit(0 if all_passed else 1)
