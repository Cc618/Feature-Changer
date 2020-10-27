import torch as T
from cc.const import path_tmp

dataset_path = path_tmp + '/celeba_faces'
img_size = 32
img_depth = 3

epochs = 0
batch_size = 128
lr = 1e-3
# TODO
z_size = 100

n_tests = 4

device = T.device('cuda' if T.cuda.is_available else 'cpu')
