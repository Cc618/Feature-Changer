#!/usr/bin/env python3

import os
import torch as T
from torchvision import transforms
try:
    import user
except:
    print('Cannot import user config (should be at src/user.py), check README')
from params import *
from net import *
from data import Dataset
from train import *
from display import *


# TODO : Move
save_path = '' # 'data/net2'
eval_ratio = 1 / 20
n_test = 190000

# Data
transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                transforms.ToTensor()])
dataset = Dataset(n_test, eval_ratio, transform)

# Net
net = Net2().to(device)

if save_path != '' and os.path.exists(save_path):
    net.load_state_dict(T.load(save_path))
    print('Loaded model at', save_path)





# Train
# train(net, 1e-3, 1, 10, dataset, save_path)
# evl(net, 0, dataset)
# tweaks = [
#         (2e-3, 1024),
#         (1e-3, 2048),
#         (1e-3, 512),
#     ]
# losses = [[2, 1, .5], [3, 2, .3], [5, 1, 2]]
# # Display losses etc
# tweaks = [
#         (2e-3, 1024),
#         (1e-3, 2048),
#         (1e-3, 512),
#     ]
# display_loss([(f'lr={lr} batch_size={bs}', loss) \
#     for (lr, bs), loss in zip(tweaks, losses)], 3)
# print(tune_stats(net, 1, tweaks, dataset))


# TODO : Move to a separate module (display)
# Test
net.eval()
dataset.mode = 'test'
testloader = T.utils.data.DataLoader(dataset, batch_size=n_tests,
        shuffle=False)
batch = next(iter(testloader))[:n_tests]
batch = batch.to(device)

display(net, batch)
