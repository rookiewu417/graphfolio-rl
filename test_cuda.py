"""Test CUDA GPU availability and performance."""
import torch

print("=" * 60)
print("CUDA GPU Test")
print("=" * 60)

# Check CUDA availability
print(f"\n1. CUDA Available: {torch.cuda.is_available()}")
print(f"2. CUDA Version: {torch.version.cuda}")
print(f"3. cuDNN Version: {torch.backends.cudnn.version()}")

# GPU info
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"4. GPU Count: {device_count}")
    
    for i in range(device_count):
        print(f"\n--- GPU {i} ---")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    
    # Simple computation test
    print(f"\n5. Running computation test...")
    device = torch.device("cuda:0")
    
    # Create large tensors
    size = 5000
    A = torch.randn(size, size, device=device)
    B = torch.randn(size, size, device=device)
    
    # Matrix multiplication (GPU intensive)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    C = A @ B
    end.record()
    
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end)
    
    print(f"   Matrix size: {size}x{size}")
    print(f"   Computation time: {elapsed:.2f} ms")
    print(f"   Result shape: {C.shape}")
    print(f"   Result device: {C.device}")
    print(f"   [OK] GPU computation successful!")
else:
    print("\n   [FAIL] No CUDA GPU available!")
    print("   Possible causes:")
    print("   - NVIDIA driver not installed")
    print("   - CUDA toolkit version mismatch")
    print("   - PyTorch CPU-only version installed")

print("\n" + "=" * 60)