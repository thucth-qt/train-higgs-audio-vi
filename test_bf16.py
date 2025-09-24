import torch

def test_bf16_support():
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))
    
    # Check framework-level BF16 support
    print("BF16 supported:", torch.cuda.is_bf16_supported())

    # Try a simple BF16 operation
    try:
        a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        c = torch.matmul(a, b)
        print("BF16 matmul success, result dtype:", c.dtype)
    except Exception as e:
        print("BF16 operation failed:", e)

if __name__ == "__main__":
    test_bf16_support()
