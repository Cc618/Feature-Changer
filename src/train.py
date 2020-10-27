import torch as T
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
# TODO : Rm
from cc.ai import DatasetIterator
from params import *


def evl(net, batch_size, dataset):
    '''
    Evaluation session
    - batch_size : If 0, takes only one batch
    - Returns the loss
    '''
    print('### Evaluation')

    opti = optim.Adam(net.parameters(), lr)
    criterion = F.mse_loss

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


def train(net, lr, epochs, batch_size, dataset, save_path):
    '''
    Training session
    - Returns losses (for each batch)
    '''
    print('### Training')

    opti = optim.Adam(net.parameters(), lr)
    criterion = F.mse_loss

    net.train(True)
    dataset.mode = 'train'
    dataloader = T.utils.data.DataLoader(dataset, batch_size=batch_size,
            shuffle=True)
    iterator = DatasetIterator(dataloader, epochs)

    losses = []
    bar = tqdm(iterator)

    for (epoch, batch), x in bar:
        x = x.to(device)
        reconstructed = net(x)

        loss = criterion(reconstructed, x)

        opti.zero_grad()
        loss.backward()
        opti.step()

        # Update metrics
        losses.append(loss)

        # TODO : Eval each epoch ?

        bar.set_postfix({'loss': loss.item()})

    if save_path != '':
        T.save(net.state_dict(), save_path)
        print('Saved model at', save_path)

    return losses
