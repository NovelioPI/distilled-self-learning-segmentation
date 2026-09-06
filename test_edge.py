import torch
import time
import os
import psutil
from pathlib import Path
import numpy as np
import cpuinfo
from thop import profile
import platform
from train import BaseModel
import json
from tqdm import tqdm

# Set seed for reproducibility
torch.manual_seed(42)


cpu_core = 4
os.environ["OMP_NUM_THREADS"] = f"{cpu_core}"
os.environ["MKL_NUM_THREADS"] = f"{cpu_core}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{cpu_core}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{cpu_core}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{cpu_core}"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_num_threads(cpu_core)
torch.set_num_interop_threads(1)

NUM_CLASSES = 19
device = torch.device('cpu')


print("=" * 40)
print("CPU Information")
print("=" * 40)
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Machine: {platform.machine()}")
print(f"Processor: {platform.processor()}")
print("PyTorch intra-op threads:", torch.get_num_threads())
print("PyTorch inter-op threads:", torch.get_num_interop_threads())

if cpuinfo:
    info = cpuinfo.get_cpu_info()
    print(f"CPU Model: {info.get('brand_raw', 'Unknown')}")
    print(f"Arch: {info.get('arch', 'Unknown')}")
    print(f"Cores: {os.cpu_count()}")
    print(f"L2 Cache Size: {info.get('l2_cache_size', 'Unknown')}")
    print(f"L3 Cache Size: {info.get('l3_cache_size', 'Unknown')}")
    print(f"Advertised Hz: {info.get('hz_advertised_friendly', 'Unknown')}")
    print(f"Actual Hz: {info.get('hz_actual_friendly', 'Unknown')}")
else:
    print("Install `py-cpuinfo` for detailed CPU info: pip install py-cpuinfo")
    print(f"Cores: {os.cpu_count()}")

if psutil:
    print(f"Logical CPUs: {psutil.cpu_count(logical=True)}")
    print(f"Physical CPUs: {psutil.cpu_count(logical=False)}")
    try:
        print(f"CPU Frequency: {psutil.cpu_freq().current:.2f} MHz")
    except Exception:
        pass
else:
    print("Install `psutil` for more CPU info: pip install psutil")

print("=" * 40)

# ---- 1. Load Model ----
root_dir = Path("/media/esr/ssd0/")
model_list_dir = [p.parent for p in (root_dir / 'saved_models').glob('**/**/*.ckpt')]
model_list_dir.sort()
save_results = []

# ---- 2. Prepare Example Input ----
input_shape = (1, 3, 256, 256)
input_tensor = torch.randn(input_shape, device=device)

for model_dir in tqdm(model_list_dir, desc="Testing models"):
    name = str(model_dir.name)
    backbone = model_dir.parent.name.split('-')[2].split('_')[0]
    
    use_kd = 'kd' in name
    if use_kd:
        parts = name.split('_')
        index = parts.index('kd')
        temperature = float(parts[index + 1][1:])
    else:
        temperature = None
        
    use_refinement = 'refinement' in name
    if use_refinement:
        entropt_thr = name.split('_')[1].split('-')[1]
        if entropt_thr == 1.0:
            entropt_thr = None
        else:
            entropt_thr = entropt_thr
    else:
        entropt_thr = None
        
        
    model = BaseModel.load_from_checkpoint(
        model_dir / 'best.ckpt',
        hparams_file=model_dir / 'hparams.yaml',
    ).to(device)
    model.eval()

    # ---- 3. Inference Time & Throughput ----
    NUM_RUNS = 20
    times = []
    with torch.no_grad():
        # Warmup
        _ = model(input_tensor)
        for _ in range(NUM_RUNS):
            start = time.time()
            _ = model(input_tensor)
            end = time.time()
            times.append(end - start)
    latency_ms = np.mean(times) * 1000
    fps = 1.0 / np.mean(times)
    # print(f"Inference time: {latency_ms:.2f} ms")
    # print(f"Throughput: {fps:.2f} FPS")

    # ---- 4. Memory Usage ----
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / 1024 ** 2
    # print(f"RAM used (process): {ram_mb:.2f} MB")

    # ---- 5. Model Size ----
    model_size_mb = os.path.getsize(model_dir / 'best.ckpt') / 1024 ** 2
    # print(f"Model size: {model_size_mb:.2f} MB")

    # ---- 6. FLOPs and Parameters ----
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    # print(f"MACs: {macs/1e6:.2f} M, Params: {params/1e6:.2f} M")
    # print(f"FLOPs: {macs*2/1e6:.2f} M")
    
    # Save to json file
    model_info = {
        "dir": model_dir.parent.name + '/' + model_dir.name,
        "backbone": backbone,
        "use_kd": use_kd,
        "T": temperature,
        "use_refinement": use_refinement,
        "entropt_thr": entropt_thr,
        "model_size_mb": model_size_mb,
        "macs": macs,
        "params": params,
        "inference_time_ms": latency_ms,
        "throughput_fps": fps,
        "ram_mb": ram_mb,
        "GLOPS": (macs * 2 / 1e9) / latency_ms * 1000,
        "cpu_core": cpu_core,
    }
    save_results.append(model_info)

# save_results.sort(key=lambda x: x['model_name'])
with open(f'model_performance_{cpu_core}.json', 'w') as f:
    json.dump(save_results, f, indent=4)