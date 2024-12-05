import numpy as np
from numba import cuda


def test_cuda():
    print("Testing CUDA availability...")

    if not cuda.is_available():
        print("CUDA is not available!")
        return False

    print(f"CUDA Device Name: {cuda.get_current_device().name}")
    print(f"Compute Capability: {cuda.get_current_device().compute_capability}")

    try:

        @cuda.jit
        def add_kernel(x, y, out):
            idx = cuda.grid(1)
            if idx < out.size:
                out[idx] = x[idx] + y[idx]

        n = 1000000
        x_host = np.arange(n, dtype=np.float32)
        y_host = np.arange(n, dtype=np.float32)

        x_device = cuda.to_device(x_host)
        y_device = cuda.to_device(y_host)
        out_device = cuda.device_array_like(x_host)

        threadsperblock = 256
        blockspergrid = (n + threadsperblock - 1) // threadsperblock
        add_kernel[blockspergrid, threadsperblock](x_device, y_device, out_device)

        out_host = out_device.copy_to_host()

        np.testing.assert_array_equal(out_host, x_host + y_host)
        print("CUDA test successful!")
        return True

    except Exception as e:
        print(f"CUDA test failed: {e}")
        return False


if __name__ == "__main__":
    test_cuda()
