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


def display_grid(grid, show=True, labels=None, title=None):
    with T.no_grad():
        f, axarr = plt.subplots(len(grid), len(grid[0]))

        # Grid display
        toplt = lambda img: img.detach().permute(1, 2, 0).cpu() \
                .squeeze().numpy()
        for i in range(len(grid)):
            ax = axarr if len(grid) == 1 else axarr[i]
            for j in range(len(grid[0])):
                ax[j].imshow(toplt(grid[i][j]).clip(0, 1),
                        vmin=0, vmax=1)

            if labels is not None:
                ax[0].set_ylabel(labels[i])

        if title is not None:
            f.suptitle(title)

        if show:
            plt.show()

        return f


def display_loss(config, show=True):
    '''
    Displays all losses given all epochs
    - config : A list of TrainingResult
    - Returns the figure
    '''
    f, ax = plt.subplots()

    for result in config:
        ax.plot([i * result.epochs / (len(result.losses) - 1) \
                for i in range(len(result.losses))],
                result.losses, label=result.name)

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    if show:
        plt.show()

    return f


# TODO : rm
if __name__ == '__main__':
    from data import TrainingResult

    display_loss([
        TrainingResult('Net', 0, 0, 3, [2, 1.8, 1.4, 1.45, 1.3, 1.26, 1.25]),
        TrainingResult('Net2', 0, 0, 2, [3, 1.9, 1.4, 1.2, 1, .26, .25])])
