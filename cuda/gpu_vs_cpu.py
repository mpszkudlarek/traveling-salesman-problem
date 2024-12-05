import time

import matplotlib.pyplot as plt
import numpy as np
from numba import cuda


def cpu_add(x, y):
    return x + y


@cuda.jit
def gpu_add_kernel(x, y, out):

    pos = cuda.grid(1)

    if pos < x.size:
        out[pos] = x[pos] + y[pos]


def gpu_add(x, y):
    out = np.zeros_like(x)

    threads_per_block = 256
    blocks_per_grid = (x.size + (threads_per_block - 1)) // threads_per_block

    x_device = cuda.to_device(x)
    y_device = cuda.to_device(y)
    out_device = cuda.to_device(out)

    gpu_add_kernel[blocks_per_grid, threads_per_block](x_device, y_device, out_device)

    return out_device.copy_to_host()


def benchmark(num_iterations=5):
    sizes = [1000, 10000, 100000, 1000000, 10000000]
    cpu_times = []
    gpu_times = []

    warmup_size = 1000000
    x_warmup = np.random.rand(warmup_size).astype(np.float32)
    y_warmup = np.random.rand(warmup_size).astype(np.float32)
    _ = gpu_add(x_warmup, y_warmup)

    for n in sizes:
        x = np.random.rand(n).astype(np.float32)
        y = np.random.rand(n).astype(np.float32)

        cpu_times_iter = []
        gpu_times_iter = []

        for _ in range(num_iterations):

            start = time.perf_counter()
            cpu_result = cpu_add(x, y)
            cpu_times_iter.append(time.perf_counter() - start)

            start = time.perf_counter()
            gpu_result = gpu_add(x, y)
            gpu_times_iter.append(time.perf_counter() - start)


            if not np.allclose(cpu_result, gpu_result, rtol=1e-5, atol=1e-5):
                print(f"Warning: Results differ for size {n}")

        cpu_time = np.median(cpu_times_iter)
        gpu_time = np.median(gpu_times_iter)

        cpu_times.append(cpu_time)
        gpu_times.append(gpu_time)

        print(f"Size {n}: CPU {cpu_time:.6f}s, GPU {gpu_time:.6f}s, " f"Speedup: {cpu_time/gpu_time:.2f}x")

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, cpu_times, "o-", label="CPU")
    plt.plot(sizes, gpu_times, "o-", label="GPU")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Array Size")
    plt.ylabel("Time (seconds)")
    plt.title("CPU vs GPU Performance Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    benchmark()
