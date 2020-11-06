import torch as T

# Progressive growing architecture
pg = True

img_size = 64 if pg else 32
img_depth = 3

# Currently not used
sparse_ratio = 1e-3

z_size = 100

device = T.device('cuda' if T.cuda.is_available else 'cpu')
