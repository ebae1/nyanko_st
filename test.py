from numba import cuda
import numpy as np

def test_cuda_to_device():
    arr = np.array([1, 2, 3], dtype=np.int32)  # 明示的にint32 np.ndarray
    assert arr.flags['C_CONTIGUOUS'], "配列はC連続である必要があります"
    cuda.select_device(0)  # 明示的GPU選択・初期化
    d_arr = cuda.to_device(arr)
    h_arr = d_arr.copy_to_host()
    print("GPU転送成功:", h_arr)

if __name__ == "__main__":
    test_cuda_to_device()
