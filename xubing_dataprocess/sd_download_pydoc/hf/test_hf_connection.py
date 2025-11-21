#!/usr/bin/env python3
"""
快速测试 Hugging Face 连接性
"""

import requests
import time

BASE = "https://huggingface.co"
DOCS_ROOT = f"{BASE}/docs"

def test_connection():
    print("=" * 60)
    print("Testing Hugging Face Connection")
    print("=" * 60)
    
    # Test 1: Base URL
    print("\n[Test 1] Testing base URL:", BASE)
    try:
        start = time.time()
        resp = requests.get(BASE, timeout=10)
        elapsed = time.time() - start
        print(f"  ✓ Status: {resp.status_code}")
        print(f"  ✓ Time: {elapsed:.2f}s")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return
    
    # Test 2: Docs root
    print(f"\n[Test 2] Testing docs root: {DOCS_ROOT}")
    try:
        start = time.time()
        resp = requests.get(DOCS_ROOT, timeout=10)
        elapsed = time.time() - start
        print(f"  ✓ Status: {resp.status_code}")
        print(f"  ✓ Time: {elapsed:.2f}s")
        print(f"  ✓ Content length: {len(resp.text)} bytes")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return
    
    # Test 3: Transformers docs
    transformers_url = f"{DOCS_ROOT}/transformers/"
    print(f"\n[Test 3] Testing transformers docs: {transformers_url}")
    try:
        start = time.time()
        resp = requests.get(transformers_url, timeout=10)
        elapsed = time.time() - start
        print(f"  ✓ Status: {resp.status_code}")
        print(f"  ✓ Time: {elapsed:.2f}s")
        print(f"  ✓ Content length: {len(resp.text)} bytes")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return
    
    # Test 4: robots.txt
    robots_url = f"{BASE}/robots.txt"
    print(f"\n[Test 4] Testing robots.txt: {robots_url}")
    try:
        start = time.time()
        resp = requests.get(robots_url, timeout=10)
        elapsed = time.time() - start
        print(f"  ✓ Status: {resp.status_code}")
        print(f"  ✓ Time: {elapsed:.2f}s")
        print(f"  ✓ Content preview:")
        lines = resp.text.split('\n')[:10]
        for line in lines:
            print(f"    {line}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return
    
    print("\n" + "=" * 60)
    print("All tests passed! Connection is working.")
    print("=" * 60)

if __name__ == "__main__":
    test_connection()

