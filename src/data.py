import pandas as pd
from PIL import Image
from torch.utils import data
from torchvision.transforms import ToTensor
from params import dataset_path

attr_path = dataset_path + '/list_attr_celeba.csv'
# TODO : Good path ?
img_path = dataset_path + '/img_align_celeba/img_align_celeba'




# # A class to create a pytorch dataset from a directory of image files

# from glob import glob
# from PIL import Image
# from torch.utils.data import Dataset
# from cc import const

# class ImageDataset(Dataset):
#     def __init__(self, images_dir, transform=None):
#         self.images_dir = images_dir
#         self.transform = transform
#         self.items = list(glob(self.images_dir + '/*'))

#     def __len__(self):
#         return len(self.items)

#     def __getitem__(self, index):
#         img = Image.open(self.items[index])

#         return img if self.transform is None else self.transform(img)



class Dataset(data.Dataset):
    '''
    Serves as a pytorch dataset and also a database to query attributes etc...
    '''
    def __init__(self, n_test, eval_ratio, transform=None):
        self.transform = ToTensor() if transform is None else transform

        self.attrs = pd.read_csv(attr_path)
        imgs = self.attrs['image_id']

        assert len(imgs) > n_test + 1, 'Not enough dataset files'

        nottest = len(imgs) - n_test
        self.test_set = imgs[:n_test].tolist()
        self.eval_set = imgs[n_test : int(nottest * eval_ratio)].tolist()
        self.train_set = imgs[int(nottest * eval_ratio):].tolist()

        # Either train, eval or test
        self.mode = 'train'

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_set)
        if self.mode == 'eval':
            return len(self.eval_set)
        return len(self.test_set)

    def __getitem__(self, index):
        if self.mode == 'train':
            items = self.train_set
        elif self.mode == 'eval':
            items = self.eval_set
        else:
            items = self.test_set

        img = Image.open(img_path + '/' + items[index])

        return self.transform(img)

    def list_attrs(self):
        '''
        Prints all attributes
        '''
        print('\n'.join(map(lambda x: '- ' + x,
            self.attrs.columns.values[1:])))

    def get_attrs(self, attr):
        '''
        Returns all images matching the attribute name (in all sets).
        '''
        return self.attrs.loc[self.attrs[attr] == 1]

    # def split_data(n_test, eval_ratio):
    #     '''
    #     Returns the list of file paths with n_tests test files, #files * eval_ratio
    #     eval files and remaining files as training set
    #     - Returns (test, eval, train)
    #     * Images are not shuffled
    #     '''
    #     df = pd.read_csv(attr_path)
    #     imgs = df['image_id']

    #     assert len(imgs) > n_test + 1, 'Not enough dataset files'

    #     nottest = len(imgs) - n_test
    #     return imgs[:n_test], imgs[n_test : int(nottest * eval_ratio)], \
    #             imgs[int(nottest * eval_ratio):]


    # def get_attr_imgs(


# TODO : Change source file
if __name__ == '__main__':
    eval_ratio = 1 / 20
    n_test = 8

    # test, evl, train = split_data(8, eval_ratio)

    # print(f'Parsed {len(train) + len(evl) + len(test)} images :')
    # print(f'{len(train):7} training images')
    # print(f'{len(evl):7} evaluation images ({eval_ratio * 100:.0f}%)')
    # print(f'{len(test):7} test images')

    dataset = Dataset(n_test, eval_ratio)

    print('### Attributes :')
    dataset.list_attrs()

    print(len(dataset.get_attrs('Smiling')), 'smiling people')
    print(len(dataset.get_attrs('Male')), 'men')

