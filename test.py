import numpy as np
from numba import cuda
import math

@cuda.jit(debug=True, opt=False)
def double_elements(arr):
    idx = cuda.grid(1)
    if idx < arr.size:
        arr[idx] *= 2


def main():
    # CUDAデバイス選択・初期化
    cuda.select_device(0)

    data = np.arange(10, dtype=np.float32)
    threads_per_block = 32
    blocks_per_grid = (data.size + threads_per_block -1) // threads_per_block

    print("元の配列:", data)

    d_data = cuda.to_device(data)
    double_elements[blocks_per_grid, threads_per_block](d_data)
    d_data.copy_to_host(data)

    print("処理後の配列:", data)

if __name__ == "__main__":
    main()
