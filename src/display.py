import torch as T
import matplotlib.pyplot as plt


def display(net, batch, show=True):
    '''
    Displays ground truth vs predicted
    - Returns the figure
    '''
    with T.no_grad():
        y = net(batch)

        f, axarr = plt.subplots(2, len(batch))

        # Grid display
        toplt = lambda img: img.detach().permute(1, 2, 0).cpu() \
                .squeeze().numpy()
        for i in range(len(batch)):
            axarr[0, i].imshow(toplt(batch[i]))
            axarr[1, i].imshow(toplt(y[i]).clip(0, 1), vmin=0, vmax=1)

        if show:
            plt.show()

        return f


def display_loss(config, epochs, show=True):
    '''
    Displays all losses given all epochs
    - config : A list of [name, losses]
    - Returns the figure
    '''
    f, ax = plt.subplots()

    for name, loss in config:
        ax.plot([i * epochs / (len(loss) - 1) for i in range(len(loss))],
                loss, label=name)

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    if show:
        plt.show()

    return f



# TODO : rm
if __name__ == '__main__':
    display_loss([
        ['Net', [2, 1.8, 1.4, 1.45, 1.3, 1.26, 1.25]],
        ['Net2', [3, 1.9, 1.4, 1.2, 1, .26, .25]],
        ], 3)
