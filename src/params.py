import torch as T

img_size = 32
img_depth = 3

# TODO : Useless ?
# epochs = 1
# batch_size = 128
# lr = 1e-3
# # TODO
z_size = 100

# n_tests = 4

device = T.device('cuda' if T.cuda.is_available else 'cpu')
