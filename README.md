# Features Changer
This ai is used to modify features in an image, for instance removing glasses.

## Structure
<!-- params update ? -->

- data : Dataset analysis
- display : Functions to plot and show data
- net : Network models
- params : Hyper parameters
- train : Training functions and statistics
- user : User config (not on git, more details bellow)

## User config
Some user specific properties are gathered within the module src/user.py.
This module is not on git, you must create it.
Here are all properties of this file :

- dataset\_path, string : Where the root of the dataset is

## Dataset
The dataset used to train the network and to make statistics is the [celeba
dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
Since it is complicated to download it via pytorch, it was downloaded from
[kaggle](https://www.kaggle.com/jessicali9530/celeba-dataset).

### Tree view
```
celeba
├── img_align_celeba
│  └── img_align_celeba
│     ├── 000001.jpg
│     └── ...
├── img_align_celeba
├── list_attr_celeba.csv
├── list_bbox_celeba.csv
├── list_eval_partition.csv
└── list_landmarks_align_celeba.csv
```
