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
from data import Dataset, TrainingResult
from train import *
from display import *
from stats import *


save_path = 'data/net3_feat'
features_path = 'data/features2'
# TODO : save_path = 'data/net2_feat'
# TODO : features_path = 'data/features3'
eval_ratio = 1 / 20
n_test = 10

# Data
transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                transforms.ToTensor()])
dataset = Dataset(n_test, eval_ratio, transform)

# Net
net = Net3().to(device)

# Load
if save_path != '' and os.path.exists(save_path):
    net.load_state_dict(T.load(save_path))
    print('Loaded model from', save_path)
else:
    # Train
    train_losses = train(net, 1e-3, 1, 256, dataset, save_path)
    print('Train losses :', train_losses)

    # loss = evl(net, 2048, dataset)
    # print('Eval loss :', loss)

# # Tweak
# results = [
#         TrainingResult('Net3,' +
#             ' lr=1e-3, batch_size=256'
#             , lr=1e-3, batch_size=256, epochs=1),
#     ]

# for r in results:
#     net = Net3().to(device)
#     r.losses = train(net, r.lr, r.epochs, r.batch_size, dataset)

# display_loss(results)




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


# Display images with same attributes
# And remove this attribute
to_rm_attr = 'Eyeglasses'
positive = True
# ratios = [1, .5, 0, -.5, -1]
ratios = [0, -1]

print(len(dataset.get_attrs(to_rm_attr, positive)), to_rm_attr, 'as', positive)
feature = gen_attrs(net, [to_rm_attr], dataset, z_size, batch_size=512)
feature = feature[to_rm_attr]

dataset.mode = 'custom'
dataset.custom_set = dataset.sample_attrs(to_rm_attr, n_test,
        positive=positive)

testloader = T.utils.data.DataLoader(dataset, batch_size=n_test,
        shuffle=False)

batch = next(iter(testloader))
batch = batch.to(device)

grid = [batch]

z = net.encode(batch)
for ratio in ratios:
    feature_changed = net.decode(z + feature * ratio)
    grid.append(feature_changed)

display_grid(grid)

exit()


# Test
net.eval()
dataset.mode = 'test'
testloader = T.utils.data.DataLoader(dataset, batch_size=n_test,
        shuffle=False)
batch = next(iter(testloader))
batch = batch.to(device)

display(net, batch)

# Generate attribute vectors
if not os.path.exists(features_path):
    # Can be chosen via dataset.list_attrs()
    # or use all_attrs = dataset.get_attr_list()
    all_attrs = [
        'Blond_Hair',
        'Eyeglasses',
        'Heavy_Makeup',
        'Male',
        'Mustache',
        'Smiling',
        'Wearing_Hat',
        'Young',
    ]
    attrs = gen_attrs(net, all_attrs, dataset, z_size, batch_size=512)
    T.save(attrs, features_path)
    print('Saved feature vectors')
else:
    attrs = T.load(features_path)
    print('Loaded feature vectors')


# Add all features within attr and show as a grid the result
net.eval()
dataset.mode = 'test'
with T.no_grad():
    testloader = T.utils.data.DataLoader(dataset, batch_size=n_test,
            shuffle=False)
    batch = next(iter(testloader))
    batch = batch.to(device)

    # Add all features
    grid = [batch]
    for category, feature in attrs.items():
        # Encode, add feature, decode
        latent = net.encode(batch)
        latent += feature
        generated = net.decode(latent)

        grid.append(generated)

    display_grid(grid)


# # Random batch
# net.eval()
# dataset.mode = 'test'
# with T.no_grad():
#     batch = T.randn([n_test, z_size], device=device)
#     generated = net.decode(batch)
#     display_grid([generated * .4])
