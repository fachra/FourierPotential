from fpm.example import mesh, axon1, axon2, plot_axons
from os.path import exists

def test_mesh():
    mesh_path = mesh()
    assert exists(mesh_path)

def test_axon1():
    points = axon1()
    assert points.shape == (2, 571)

def test_axon2():
    points = axon2()
    assert points.shape == (2, 709)

def test_plot_axons():
    plt_axons = plot_axons()
    assert plt_axons.get_size() == (1032, 1136)
