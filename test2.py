from numba import cuda

print(cuda.gpus)  # GPU一覧表示
cuda.select_device(0)  # GPU 0を選択して初期化
