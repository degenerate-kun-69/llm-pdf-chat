import torch

def bytes_to_gb(x):
    return round(x / (1024 ** 3), 2)

if not torch.cuda.is_available():
    print("❌ ROCm GPU not available.")
    exit()

device = torch.device("cuda")
print("✅ ROCm GPU detected:", torch.cuda.get_device_name(0))

# Target: ~14 GB
target_bytes = 14 * 1024 ** 3  # 14 GB
tensor_size = 1024 * 1024 * 256  # 256 MB tensors
tensors = []
allocated = 0

try:
    while allocated < target_bytes:
        t = torch.randn(tensor_size // 4, dtype=torch.float32, device=device)
        tensors.append(t)
        allocated += t.element_size() * t.nelement()
        print(f"Allocated: {bytes_to_gb(allocated)} GB")

    print("✅ Successfully allocated ~14 GB of GPU memory.")

except RuntimeError as e:
    print("❌ Allocation failed:", e)

# Optional: Keep GPU busy for a second
torch.cuda.synchronize()
input("Press Enter to release GPU memory...")
