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
from data import *
from train import *
from display import *
from stats import *


# Configure what to show
config = [
        # * Benchmark multiple network architectures
        # 'tune',
        # * Display batch
        # 'display',
        # * Save media of README in the res folder
        'media',
        # * Change one feature and display multiple levels
        # * of adding / removing it
        # 'change_feat',
        # * Show multiple images with added features on them
        # 'all_feat'
    ]

# Params
if pg:
    save_path = 'data/pg'
    features_path = 'data/features_pg'
else:
    save_path = 'data/net3_feat'
    features_path = 'data/features2'

eval_ratio = 1 / 20
n_test = 8


# Data
transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                transforms.ToTensor()])
dataset = Dataset(n_test, eval_ratio, transform)

# --- Load / Train ---
# PG hyper params
steps = 4
chan = 8

train_epochs = 1 if not pg else [
        1,
        # With merging
        1,
        1,
        1,
        # Without merging (full size)
        2
    ]

net = (PGAE(steps, chan) if pg else Net3()).to(device)
print(f'Network with {(n_weights(net) + 999) // 1000}K weights')

if pg:
    print('Using progressive growing architecture')
else:
    print('Using deep convolutional architecture')

if os.path.exists(save_path):
    net.load_state_dict(T.load(save_path))
    print('Loaded network at', save_path)

    # Final step if loaded
    if pg:
        net.step = steps
else:
    # Train
    if pg:
        train_losses = train_pg(net, 1e-3, train_epochs, 3, 128,
                dataset, save_path)
    else:
        train_losses = train(net, 1e-3, train_epochs, 256, dataset, save_path)

    # Evaluate
    if not pg:
        loss = evl(net, 2048, dataset)
        print('Eval loss :', loss)

    # We can display losses also via display_losses

# --- Tuning ---
if 'tune' in config:
    print('> Tune')

    assert not pg, 'Tuning is not available with ' + \
            'progressive growing architectures'

    results = [
            TrainingResult('Net3,' +
                ' lr=1e-3, batch_size=256'
                , lr=1e-3, batch_size=256, epochs=1),
            TrainingResult('Net3,' +
                ' lr=1e-3, batch_size=512'
                , lr=1e-3, batch_size=512, epochs=1),
        ]

    for r in results:
        net = Net3().to(device)
        r.losses = train(net, r.lr, r.epochs, r.batch_size, dataset)

    display_loss(results)


# --- Display ---
if 'display' in config:
    print('> Display')

    # Test on batch
    net.eval()
    dataset.mode = 'test'
    testloader = T.utils.data.DataLoader(dataset, batch_size=n_test,
            shuffle=False)
    batch = next(iter(testloader))
    batch = batch.to(device)

    display(net, batch)

# --- README media ---
if 'media' in config:
    print('> Media')

    def save_gif(batch, path, process=None, duration=100):
        '''
        Batch to gif
        '''
        batch = batch.clamp(0, 1).cpu()
        to_img = transforms.ToPILImage() if process is None else \
            transforms.Compose([transforms.ToPILImage(), process])
        imgs = [to_img(batch[i]) for i in range(len(batch))]
        imgs[0].save(path, save_all=True, append_images=imgs[1:], loop=0,
                duration=duration)

    def z_lerp(z_start, z_end, ratio):
        '''
        Linear interpolation from z_start to z_end
        '''
        return z_start * (1 - ratio) + z_end * ratio

    dataset.mode = 'test'
    loader = T.utils.data.DataLoader(dataset, batch_size=n_test, shuffle=False)

    net.eval()
    with T.no_grad():
        # TODO : Custom images
        batch = next(iter(loader)).to(device)

        woman_blond = batch[0]
        woman_smile = batch[1]
        man_brown = batch[6]
        z_woman_blond = net.encode(woman_blond.unsqueeze(0))
        z_woman_smile = net.encode(woman_smile.unsqueeze(0))
        z_man_brown = net.encode(man_brown.unsqueeze(0))

        # Find feature vectors
        attrs = ['Smiling', 'Male', 'Mustache', 'Blond_Hair']
        features = gen_attrs(net, attrs,
                dataset, batch_size=512)
        z_smile = features['Smiling']
        z_male = features['Male']
        z_mustache = features['Mustache']
        z_blond = features['Blond_Hair']

        # GIFs
        res = 256
        n_steps = 20
        transitions = [
                (z_woman_smile, z_woman_smile - z_smile * 2),
                (z_woman_blond, z_woman_blond - z_blond * 1.5),
                (z_woman_smile, z_woman_smile + z_male * 2),
                (z_woman_blond, z_woman_blond + z_mustache * 2),
                (z_man_brown, z_man_brown + z_mustache * 2.5),
                (z_man_brown, z_man_brown + z_blond * 1.5),
                (z_man_brown, z_man_brown - z_male * 2),
                (z_man_brown, z_woman_blond),
            ]


        # Latent lerp between each pair
        for i, (z_start, z_end) in enumerate(transitions):
            zs = T.stack([z_lerp(z_start, z_end, i / (n_steps - 1))
                        for i in range(n_steps)]).squeeze(1)

            imgs = net.decode(zs)
            save_gif(imgs, f'res/lerp_{i}.gif',
                    process=transforms.Resize((res, res)))

    print('Saved media in res')

# --- Change features ---
if 'change_feat' in config:
    print('> Change features')

    # IMG : sess2/add_{glasses,blond_hair}.png
    # Display images with same attributes
    # And remove this attribute
    to_rm_attr = 'Smiling'
    positive = False
    # ratios = [1, .5, 0, -.5, -1]
    ratios = [1, 2, 4]
    # ratios = [-1, -2, -4]

    print(len(dataset.get_attrs(to_rm_attr, positive)),
        to_rm_attr, 'as', positive)
    feature = gen_attrs(net, [to_rm_attr], dataset, batch_size=512)
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

# --- All features ---
if 'all_feat' in config:
    print('> All features')

    # Generate multiple attribute vectors
    if not os.path.exists(features_path):
        # Can be chosen via dataset.list_attrs()
        # or use all_attrs = dataset.get_attr_list()
        all_attrs = [
            'Blond_Hair',
            'Eyeglasses',
            # 'Heavy_Makeup',
            # 'Male',
            'Mustache',
            'Smiling',
            # 'Wearing_Hat',
            # 'Young',
        ]
        attrs = gen_attrs(net, all_attrs, dataset, batch_size=512)
        T.save(attrs, features_path)
        print('Saved feature vectors')
    else:
        attrs = T.load(features_path)
        print('Loaded feature vectors')

    # Big grid display
    # Show attributes
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
