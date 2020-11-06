import torch as T
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
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

        # We can also add a sparse loss to keep z normalized
        loss = criterion(reconstructed, x) # + sparse_ratio * sparse_loss(z)

        opti.zero_grad()
        loss.backward()
        opti.step()

        # Update metrics
        losses.append(loss)

        bar.set_postfix({'loss': loss.item()})

        if batch == 0 and epoch != 0 and save_path != '':
            T.save(net.state_dict(), save_path)
            print('Saved model at', save_path)

    if save_path != '':
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


def gen_pg_transform(img_size, steps, step):
    '''
    Generates a transform for a progressive growing model
    - steps : Number of times we add layers to the PGAE
    - step : Current step, 1 = No layer added, steps = All layers added
    '''
    size = img_size // 2 ** (steps - step)

    return transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor()])


def train_pg(net, lr, epochs, feature_loss_step, batch_size, dataset,
        save_path=''):
    '''
    - epochs : List of all epochs for each step + final training
            when merge_ratio is 1
    - feature_loss_step : From which step the deep feature consistent loss
            is used
    - Returns losses for each batch
    '''
    print(f'Training progressive growing model, image_size={img_size}')
    print(f'lr={lr}, batch_size={batch_size}, save_path={save_path}')
    print('Epochs :', epochs)

    stats = []

    # TODO : Save
    dataset.mode = 'train'
    net.train(True)
    loader = T.utils.data.DataLoader(dataset, batch_size=batch_size,
            shuffle=True)
    opti = optim.Adam(net.parameters(), lr)

    criterion_small = F.mse_loss
    criterion_big = FLPLoss('vae-123', device, 'mean')
    net.step = 1
    n_batch = len(dataset) // batch_size + 1
    for i in range(net.steps + 1):
        dataset.transform = gen_pg_transform(img_size, net.steps, net.step)

        criterion = criterion_big if net.step >= feature_loss_step else \
                criterion_small

        n_epoch = epochs[i]
        iterator = DatasetIterator(loader, n_epoch)
        bar = tqdm(iterator)
        for (e, b), batch in bar:
            net.merge_ratio = 1 if i >= net.steps else \
                    (e + (b + 1) / n_batch) / n_epoch

            batch = batch.to(device)
            generated = net(batch)

            loss = criterion(generated, batch)
            stats.append(loss.item())

            opti.zero_grad()
            loss.backward()
            opti.step()

            bar.set_postfix({
                    'out_size': img_size // 2 ** (net.steps - net.step),
                    'epoch': e + 1,
                    'merge_ratio': net.merge_ratio,
                    'batch': b + 1,
                    'loss': loss.item(),
                })

            if b == n_batch - 1 and save_path != '':
                T.save(net.state_dict(), save_path)

        # print('Trained for size', img_size // 2 ** (net.steps - net.step))
        if net.step < net.steps:
            net.step += 1

    return stats
