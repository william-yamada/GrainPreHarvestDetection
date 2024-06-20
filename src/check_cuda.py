import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    # Number of GPUs available
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs Available: {num_gpus}")

    # CUDA version
    cuda_version = torch.version.cuda
    print(f"CUDA Version: {cuda_version}")

    # Current GPU device
    current_device = torch.cuda.current_device()
    print(f"Current GPU Device: {current_device}")

    # Device properties for each GPU
    for device in range(num_gpus):
        device_properties = torch.cuda.get_device_properties(device)
        print(f"Device {device} Properties: {device_properties}")

else:
    print("CUDA is not available. Running on CPU.")