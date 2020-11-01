import torch as T

img_size = 32
img_depth = 3

sparse_ratio = 1e-2

z_size = 100

device = T.device('cuda' if T.cuda.is_available else 'cpu')
