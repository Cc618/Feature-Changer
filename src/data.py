from random import shuffle
import pickle
import pandas as pd
from PIL import Image
from torch.utils import data
from torchvision.transforms import ToTensor
from user import dataset_path

attr_path = dataset_path + '/list_attr_celeba.csv'
img_path = dataset_path + '/img_align_celeba/img_align_celeba'

# Number of samples we take at random  to generate the absolute vector
attr_n_sample = 1000


def n_weights(module):
    def prod(shape):
        p = 1
        for s in shape:
            p *= s

        return p

    total = 0
    for p in module.parameters():
        total += prod(p.shape)

    return total


class DatasetIterator:
    '''
    Used to have a nice progress bar with tqdm
    Yields batches from loader "epochs" times
    '''
    def __init__(self, loader, epochs):
        self.loader = loader
        self.epochs = epochs

    def __len__(self):
        return len(self.loader) * self.epochs

    def __iter__(self):
        '''
        Returns (epoch, batch), data
        '''
        for e in range(self.epochs):
            for batch, data in enumerate(self.loader):
                yield (e, batch), data


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
        self.eval_set = imgs[n_test : n_test + int(nottest * eval_ratio)] \
                .tolist()
        self.train_set = imgs[n_test + int(nottest * eval_ratio):].tolist()
        self.custom_set = []

        # Either train, eval, test or custom
        self.mode = 'train'

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_set)
        if self.mode == 'eval':
            return len(self.eval_set)
        if self.mode == 'custom':
            return len(self.custom_set)
        return len(self.test_set)

    def __getitem__(self, index):
        if self.mode == 'train':
            items = self.train_set
        elif self.mode == 'eval':
            items = self.eval_set
        elif self.mode == 'custom':
            items = self.custom_set
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

    def get_attr_list(self):
        '''
        Get all attributes (categories)
        '''
        return self.attrs.columns.values[1:].tolist()

    def get_attrs(self, attr, positive=True, sample=False):
        '''
        Returns all images matching the attribute name (in all sets).
        '''
        if sample:
            return self.sample_attrs(attr, attr_n_sample, positive=positive)

        attrs = self.attrs.loc[self.attrs[attr] == (1 if positive else -1)]
        attrs = attrs['image_id'].tolist()

        return attrs

    def sample_attrs(self, attr, count, positive=True):
        '''
        Samples a batch of file paths containing this attribute
        '''
        attrs = self.attrs.loc[self.attrs[attr] == (1 if positive else -1)]
        attrs = attrs['image_id'].tolist()

        shuffle(attrs)
        return attrs[:count]

    def load_attrs(self, attr, positive):
        '''
        Loads all attributes within the custom set
        '''
        self.mode = 'custom'
        self.custom_set = self.get_attrs(attr, positive=positive, sample=True)

    def set_custom_images(self, names):
        '''
        Sets the custom set of images
        '''
        self.mode = 'custom'
        self.custom_set = names


class TrainingResult:
    '''
    Used to store training information useful to establish stats of a training
    session
    '''
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __init__(self, name, lr, batch_size, epochs, losses=[]):
        self.name = name
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.losses = losses

    def __repr__(self):
        return f'{self.name} : lr={self.lr}, batch_size={self.batch_size}' + \
            f', epochs={self.epochs}, last_loss={self.losses[-1]}'

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)


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

