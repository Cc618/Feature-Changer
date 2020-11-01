import torch as T
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
from params import *
from data import DatasetIterator, TrainingResult
from vgg_loss import FLPLoss


def evl(net, batch_size, dataset):
    '''
    Evaluation session
    - batch_size : If 0, takes only one batch
    - Returns the loss
    '''
    print('### Evaluation')

    criterion = FLPLoss('vae-123', device, 'mean')

    net.train(False)
    dataset.mode = 'eval'
    dataloader = T.utils.data.DataLoader(dataset,
            batch_size=len(dataset) if batch_size == 0 else batch_size,
            shuffle=True)
    iterator = DatasetIterator(dataloader, 1)

    bar = tqdm(iterator)
    loss = 0

    with T.no_grad():
        for (epoch, batch), x in bar:
            x = x.to(device)
            reconstructed = net(x)

            loss += criterion(reconstructed, x).item()
            bar.set_postfix({'loss': loss / (batch + 1)})

    return loss / (batch + 1)


def sparse_loss(x):
    '''
    loss = mean(min(abs(x), sqrt(abs(x))))
    '''
    x = x.abs().view(1, -1, z_size)
    both = T.cat([x, x.sqrt()], dim=0)

    return both.min(0)[0].mean()


def train(net, lr, epochs, batch_size, dataset, save_path=''):
    '''
    Training session
    - Returns losses (for each batch)
    '''
    print('### Training')

    opti = optim.Adam(net.parameters(), lr)
    # criterion = F.mse_loss
    criterion = FLPLoss('vae-123', device, 'mean')
    sparse_ratio = 1e-3

    net.train(True)
    dataset.mode = 'train'
    dataloader = T.utils.data.DataLoader(dataset, batch_size=batch_size,
            shuffle=True)
    iterator = DatasetIterator(dataloader, epochs)

    losses = []
    bar = tqdm(iterator)

    for (epoch, batch), x in bar:
        x = x.to(device)
        z = net.encode(x)
        reconstructed = net.decode(z)

        loss = criterion(reconstructed, x) + sparse_ratio * sparse_loss(z)

        opti.zero_grad()
        loss.backward()
        opti.step()

        # Update metrics
        losses.append(loss)

        bar.set_postfix({'loss': loss.item()})

        if batch == 0 and epoch != 0 and save_path != '':
            T.save(net.state_dict(), save_path)
            print('Saved model at', save_path)

    T.save(net.state_dict(), save_path)
    print('Saved model at', save_path)

    return losses


def tune_stats(net, epochs, hyperparams, dataset, eval_batch_size=0):
    '''
    Returns the list of statistics for each configuration
    - hyperparams : List of (lr, batch_size)
    - Returns a list for each config of evaluation error
    '''
    # TODO : Add time
    stats = []
    for i, (lr, batch_size) in enumerate(hyperparams):
        print(f'## Config {i}')

        train(net, lr, epochs, batch_size, dataset)
        loss = evl(net, eval_batch_size, dataset)

        print('> Eval loss :', loss)
        stats.append(loss)

    return stats
