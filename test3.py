from numba import cuda
import numpy as np

print(cuda.gpus)  # GPUリスト表示

def test():
    cuda.select_device(0)
    data = np.arange(10, dtype=np.float32)
    d_data = cuda.to_device(data)
    print("GPUへ転送成功")

test()
