import torch as T
import matplotlib.pyplot as plt


def display(net, batch, show=True):
    '''
    Display ground truth vs predicted
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
