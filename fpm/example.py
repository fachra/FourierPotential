"""The module provides some geometry examples."""

import numpy as np
import pkg_resources as pkr


def mesh():
    return pkr.resource_filename(__name__, 'data/example_3Dmesh.stl')


def axon1():
    filepath = pkr.resource_filename(__name__, 'data/axons/axon1.npy')
    return np.load(filepath)


def axon2():
    filepath = pkr.resource_filename(__name__, 'data/axons/axon2.npy')
    return np.load(filepath)


def plot_axons():
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    filepath = pkr.resource_filename(__name__, 'data/axons/axons_microscopy.jpg')
    img = mpimg.imread(filepath)
    imgplot = plt.imshow(img)
    plt.axis('off')

    return imgplot
