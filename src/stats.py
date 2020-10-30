import torch as T
from params import device


def absolute_attr_vector(net, attr, positive, dataset, z_size, batch_size=0):
    '''
    Returns the mean of all encoded vectors representing images
    with this attribute.
    '''
    net.train(False)
    with T.no_grad():
        latent = T.zeros([z_size], device=device)

        dataset.load_attrs(attr, positive)
        bs = len(dataset) if batch_size == 0 else batch_size
        loader = T.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False)

        for x in loader:
            x = x.to(device)
            latent += net.encode(x).sum(0) / len(dataset)

        return latent


def attr_vector(net, attr, dataset, z_size, batch_size=0):
    '''
    Get latent vector representive an attribute
    z_attr = z_attr_pos - z_attr_neg
    '''
    return absolute_attr_vector(net, attr, True, dataset, z_size, batch_size) \
            - absolute_attr_vector(net, attr, False,
                    dataset, z_size, batch_size)


def gen_attrs(net, attrs, dataset, z_size, batch_size=0):
    '''
    Returns a dict with all (additive) feature vectors
    '''
    results = {}
    for attr in attrs:
        print('Finding', attr, 'feature vector')
        results[attr] = attr_vector(net, attr, dataset, z_size, batch_size)

    return results
