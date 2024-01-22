import torch
import platform

def get_device(target='cpu'):
    # Check for device
    device_name = platform.processor()
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    device_name = line.split(":")[1].strip()
    except FileNotFoundError:
        pass
    device = 'cpu'
    if target != 'cpu' and torch.cuda.is_available():
        device = torch.device(target)
        device_name = torch.cuda.get_device_name()

    print('--- Device ---')
    print('Type:\t', str(device).upper())
    print('Name:\t', device_name)
    print()

    return device, device_name
