#!/usr/bin/env python3

import os
import torch as T
from torchvision import transforms
import matplotlib.pyplot as plt
try:
    import user
except:
    print('Cannot import user config (should be at src/user.py), check README')
from params import *
from net import *
from data import Dataset
from train import *


# TODO : Move
def display(net, batch):
    with T.no_grad():
        y = net(batch)

        _, axarr = plt.subplots(2, len(batch))

        # Grid display
        toplt = lambda img: img.detach().permute(1, 2, 0).cpu() \
                .squeeze().numpy()
        for i in range(len(batch)):
            axarr[0, i].imshow(toplt(batch[i]))
            axarr[1, i].imshow(toplt(y[i]).clip(0, 1), vmin=0, vmax=1)

        plt.show()


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
# print(tune_stats(net, 1, tweaks, dataset))



# # Test
# net.eval()
# dataset.mode = 'test'
# testloader = T.utils.data.DataLoader(dataset, batch_size=n_tests,
#         shuffle=False)
# # batch, _ = next(iter(testloader))[:n_tests]
# batch = next(iter(testloader))[:n_tests]
# print(batch.shape)
# batch = batch.to(device)
# display(net, batch)

