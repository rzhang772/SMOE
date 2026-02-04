import time
import csv
import psutil
from pynvml import *

# ===== 初始化 =====
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(3)
process = psutil.Process()

output_file = f"./expirments/system_util_trace_{time.time()}.csv"

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp",
        "gpu_util",          # %
        "gpu_mem_ctrl_util",      # %
        "gpu_mem_used_MB", # memory used in MB
        "gpu_mem_util_pct", # memory utilization in %
        "pcie_tx_MBps",
        "pcie_rx_MBps",
        "cpu_total_util",    # %
        "cpu_proc_util"      # %
    ])

    # psutil 第一次调用用于 warm-up
    psutil.cpu_percent(interval=None)
    process.cpu_percent(interval=None)

    start = time.time()
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Starting system utilization tracing...")
    while time.time() - start < 60:
        ts = time.time()

        # GPU
        util = nvmlDeviceGetUtilizationRates(handle)

        mem = nvmlDeviceGetMemoryInfo(handle)
        mem_used_MB  = mem.used / 1024 / 1024
        mem_total_MB = mem.total / 1024 / 1024
        mem_util_pct = mem_used_MB / mem_total_MB * 100

        pcie_tx = nvmlDeviceGetPcieThroughput(
            handle, NVML_PCIE_UTIL_TX_BYTES
        ) / 1024
        pcie_rx = nvmlDeviceGetPcieThroughput(
            handle, NVML_PCIE_UTIL_RX_BYTES
        ) / 1024

        # CPU
        cpu_total = psutil.cpu_percent(interval=None)
        cpu_proc = process.cpu_percent(interval=None)

        writer.writerow([
            ts,
            util.gpu,
            util.memory,
            mem_used_MB,
            mem_util_pct,
            pcie_tx,
            pcie_rx,
            cpu_total,
            cpu_proc
        ])

        time.sleep(0.1)  # 10 Hz