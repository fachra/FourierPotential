import pytest
import numpy as np
from fpm import Circle, Ellipse, Shape, project_stl
from fpm.example import mesh, axon1, axon2
from numpy.testing import assert_almost_equal


# Test class Circle.

def test_circle_center_x_is_not_numeric():
    with pytest.raises(ValueError):
        c = Circle('10.2', 0, 1)


def test_circle_center_y_is_not_numeric():
    with pytest.raises(ValueError):
        c = Circle(10.2, '0.0', 1)


def test_circle_radius_is_not_numeric():
    with pytest.raises(ValueError):
        c = Circle(10.2, 0, '1')


def test_circle_positional_argument_missing_1():
    with pytest.raises(TypeError):
        c = Circle()


def test_circle_positional_argument_missing_2():
    with pytest.raises(TypeError):
        c = Circle(0, 0)


def test_circle_radius_is_negative():
    with pytest.raises(ValueError):
        c = Circle(10.2, 0, -1)


def test_circle_n_points_is_not_integer_1():
    with pytest.raises(ValueError):
        c = Circle(0, 0, 1, n_points='10.2')


def test_circle_n_points_is_not_integer_2():
    with pytest.raises(ValueError):
        c = Circle(0, 0, 1, n_points=10.2)


def test_circle_n_points_is_too_small():
    with pytest.raises(ValueError):
        c = Circle(0, 0, 1, n_points=1)


def test_circle_maximum_distance_is_not_numeric():
    with pytest.raises(ValueError):
        c = Circle(0, 0, 1, dist_max='0.002')


def test_circle_maximum_distance_is_negative():
    with pytest.raises(ValueError):
        c = Circle(0, 0, 1, dist_max=0)


def test_circle_center_x_coordinate():
    c = Circle(10, 23.5, 1)
    assert_almost_equal(c.center_x, 10)


def test_circle_center_y_coordinate():
    c = Circle(10, -23.5, 1)
    assert_almost_equal(c.center_y, -23.5)


def test_circle_radius():
    c = Circle(0, 0, 1)
    assert_almost_equal(c.r, 1)


def test_circle_perimeter():
    c = Circle(0, 0, 1)
    assert_almost_equal(c.perimeter, 2*np.pi)


def test_circle_area():
    c = Circle(0, 0, 1)
    assert_almost_equal(c.area, np.pi)


def test_circle_number_of_points_1():
    c = Circle(0, 0, 1, n_points=100)
    assert c.n_points == 100


def test_circle_number_of_points_2():
    c = Circle(0, 0, 1, n_points=100)
    assert len(c) == 100


def test_circle_maximum_distance():
    c = Circle(0, 0, 1, dist_max=0.1)
    assert np.max(c.dl) <= 0.1


def test_circle_curvature():
    c = Circle(0, 0, 2, n_points=100)
    curva = np.ones(100) / 2
    assert_almost_equal(c.curvature, curva)


def test_circle_assert_outpointing_normals():
    c = Circle(11.3, -12.9, 2)
    points = c.points - c.shift
    normals = c.normals
    cross_product = np.sum(points * normals, axis=0)
    assert np.all(cross_product >= 0)


def test_circle_plot():
    from matplotlib.lines import Line2D
    c = Circle(11.3, -12.9, 2)
    h = c.plot()
    assert isinstance(h[0], Line2D)


def test_circle_repr():
    c = Circle(10.2, 7.8, 2, n_points=100)
    repr_str = ('<Circle(Ellipse)>: center_x = 10.2,\n'
                '                   center_y = 7.8,\n'
                '                   r = 2,\n'
                '                   n_points = 100.')
    assert repr(c) == repr_str


def test_circle_str():
    c = Circle(10.2, 7.8, 2, n_points=100)
    s = ('A circle with\n'
         '    radius = 2,\n'
         '    centered at (10.2, 7.8).')
    assert str(c) == s


# Test class Ellipse.


def test_ellipse_center_x_is_not_numeric_1():
    with pytest.raises(ValueError):
        e = Ellipse('10.2', 0, 1, 2)


def test_ellipse_center_x_is_not_numeric_2():
    with pytest.raises(ValueError):
        e = Ellipse(np.inf, 0, 1, 2)


def test_ellipse_center_y_is_not_numeric():
    with pytest.raises(ValueError):
        e = Ellipse(10.2, '0.0', 1, 2)


def test_ellipse_semi_major_axis_a_is_not_numeric():
    with pytest.raises(ValueError):
        e = Ellipse(0, 0, '1', 2)


def test_ellipse_semi_major_axis_a_is_negative():
    with pytest.raises(ValueError):
        e = Ellipse(0, 0, -1, 2)


def test_ellipse_semi_minor_axis_b_is_not_numeric():
    with pytest.raises(ValueError):
        e = Ellipse(0, 0, 1, '2')


def test_ellipse_semi_minor_axis_b_is_negative():
    with pytest.raises(ValueError):
        e = Ellipse(0, 0, 1, -2)


def test_ellipse_n_points_is_not_integer_1():
    with pytest.raises(ValueError):
        e = Ellipse(0, 0, 1, 2, n_points='10.2')


def test_ellipse_n_points_is_not_integer_2():
    with pytest.raises(ValueError):
        e = Ellipse(0, 0, 1, 2, n_points=10.2)


def test_ellipse_n_points_is_too_small():
    with pytest.raises(ValueError):
        e = Ellipse(0, 0, 1, 2, n_points=1)


def test_ellipse_maximum_distance_is_not_numeric():
    with pytest.raises(ValueError):
        e = Ellipse(0, 0, 1, 2, dist_max='0.002')


def test_ellipse_maximum_distance_is_negative():
    with pytest.raises(ValueError):
        e = Ellipse(0, 0, 1, 2, dist_max=0)


def test_ellipse_rotate_angle_is_not_numeric():
    with pytest.raises(ValueError):
        e = Ellipse(0, 0, 1, 2, rotate_angle='90')


def test_ellipse_positional_argument_missing_1():
    with pytest.raises(TypeError):
        e = Ellipse()


def test_ellipse_positional_argument_missing_2():
    with pytest.raises(TypeError):
        e = Ellipse(0, 0)


def test_ellipse_positional_argument_missing_3():
    with pytest.raises(TypeError):
        e = Ellipse(0, 0, 1)


def test_ellipse_center_x_coordinate():
    e = Ellipse(10.2, 20.3, 1, 2)
    assert_almost_equal(e.center_x, 10.2)


def test_ellipse_center_y_coordinate():
    e = Ellipse(10.2, 20.3, 1, 2)
    assert_almost_equal(e.center_y, 20.3)


def test_ellipse_semi_major_axis():
    e = Ellipse(10.2, 20.3, 1, 2)
    assert_almost_equal(e.a, 1)


def test_ellipse_semi_minor_axis():
    e = Ellipse(10.2, 20.3, 1, 2)
    assert_almost_equal(e.b, 2)


def test_ellipse_number_of_points_1():
    e = Ellipse(0, 0, 1, 2, n_points=100, refine=False)
    assert_almost_equal(e.n_points, 100)


def test_ellipse_number_of_points_2():
    e = Ellipse(0, 0, 1, 2, n_points=100, refine=False)
    assert len(e) == 100


def test_ellipse_maximum_distance():
    e = Ellipse(0, 0, 1, 2, dist_max=0.1)
    assert np.max(e.dl) <= 0.1


def test_ellipse_rotate_angle_1():
    e1 = Ellipse(10, -20.3, 1, 2, rotate_angle=60)
    e2 = Ellipse(10, -20.3, 1, 2, rotate_angle=-300)
    assert_almost_equal(e1.points, e2.points)
    assert_almost_equal(e1.normals, e2.normals)


def test_ellipse_rotate_angle_2():
    e = Ellipse(0, 0, 1, 2,
                rotate_angle=90, refine=False, n_points=4)
    desired_normals = np.array(
        [[0, -1, 0, 1],
         [1, 0, -1, 0]]
    )
    assert_almost_equal(e.normals, desired_normals)


def test_ellipse_refine_1():
    e1 = Ellipse(0, 0, 10, 10.1, refine=False)
    e2 = Ellipse(0, 0, 10, 10.1, refine=True)
    assert len(e1) <= len(e2)


def test_ellipse_refine_2():
    e1 = Ellipse(0, 0, 1, 2, refine=False)
    e2 = Ellipse(0, 0, 1, 2, refine=True)
    assert len(e1) < len(e2)


def test_ellipse_refine_3():
    e1 = Ellipse(0, 0, 10, 1, refine=False)
    e2 = Ellipse(0, 0, 10, 1, refine=True)
    assert len(e1) < len(e2)


def test_ellipse_points():
    e = Ellipse(0, 0, 2, 1, n_points=4, refine=False)
    desired_points = np.array(
        [[2, 0, -2, 0],
         [0, 1, 0, -1]]
    )
    assert_almost_equal(e.points, desired_points)


def test_ellipse_curvature():
    e = Ellipse(0, 0, 2, 1, n_points=4, refine=False)
    desired_curvature = np.array(
        [2, 1/4, 2, 1/4]
    )
    assert_almost_equal(e.curvature, desired_curvature)


def test_ellipse_normals():
    e = Ellipse(0, 0, 2, 1, n_points=4, refine=False)
    desired_normals = np.array(
        [[1, 0, -1, 0],
         [0, 1, 0, -1]]
    )
    assert_almost_equal(e.normals, desired_normals)


def test_ellipse_assert_outpointing_normals():
    e = Ellipse(12.3, 31.4, 2, 1)
    points = e.points - np.array([[12.3], [31.4]])
    normals = e.normals
    cross_product = np.sum(points * normals, axis=0)
    assert np.all(cross_product >= 0)


def test_ellipse_plot():
    from matplotlib.lines import Line2D
    e = Ellipse(12.3, 31.4, 2, 1)
    h = e.plot()
    assert isinstance(h[0], Line2D)


def test_ellipse_perimeter():
    e = Ellipse(0, 0, 2, 1)
    desired_perimeter = 9.68845
    assert_almost_equal(e.perimeter, desired_perimeter, decimal=4)


def test_ellipse_area():
    e = Ellipse(0, 0, 2, 1)
    desired_area = np.pi*2
    assert_almost_equal(e.area, desired_area)


def test_ellipse_repr():
    e = Ellipse(10.2, -7.8, 2, 1,
                n_points=100, refine=False, rotate_angle=60)
    repr_str = ('<Ellipse>: center_x = 10.2,\n'
                '           center_y = -7.8,\n'
                '           a = 2, b = 1,\n'
                '           n_points = 100,\n'
                '           rotate_angle = 60.')
    assert repr(e) == repr_str


def test_ellipse_str():
    e = Ellipse(10.2, -7.8, 2, 1,
                n_points=100, refine=False, rotate_angle=60)
    s = ('An ellipse with\n'
         '    a = 2, b = 1,\n'
         '    centered at (10.2, -7.8),\n'
         '    rotated 60 degrees\n'
         '    in anticlockwise direction.')
    assert str(e) == s


# Test class Shape.


def test_shape_model_is_polygon():
    poly = project_stl(mesh())
    shp = Shape(poly)
    error = np.abs(shp.area - poly.area) / poly.area
    assert error < 0.1


def test_shape_model_is_ndarray_1():
    shp = Shape(axon1())
    desired_area = 50.97855954997717
    assert_almost_equal(shp.area, desired_area)


def test_shape_model_is_ndarray_2():
    shp = Shape(axon2())
    desired_area = 91.4027008549474
    assert_almost_equal(shp.area, desired_area)


def test_shape_model_is_ndarray_counterclockwise_winding():
    theta = np.linspace(0, 2*np.pi, 100)
    points = np.c_[np.cos(theta), np.sin(theta)].T
    shp = Shape(points)
    assert_almost_equal(shp.area, np.pi)


def test_shape_model_is_ndarray_clockwise_winding():
    theta = np.linspace(0, 2*np.pi, 100)
    points = np.c_[np.cos(theta), np.sin(theta)].T
    shp = Shape(points[:, ::-1])
    assert_almost_equal(shp.area, np.pi)


def test_shape_model_is_other_object():
    with pytest.raises(NotImplementedError):
        shp = Shape([[1, 2, 3], [4, 5, 6]])


def test_shape_model_wrong_shape_1():
    model = np.array(
        [[1, 0, -1, 0],
         [0, 1, 0, -1],
         [0, 0, 0, 0]]
    )
    with pytest.raises(ValueError):
        shp = Shape(model)


def test_shape_model_wrong_shape_2():
    model = np.array(
        [[1, 0, -1, 0],
         [0, 1, 0, -1]]
    ).T
    with pytest.raises(ValueError):
        shp = Shape(model)


def test_shape_model_wrong_shape_3():
    model = np.array(
        [[1, 0],
         [0, 1]]
    )
    with pytest.raises(ValueError):
        shp = Shape(model)


def test_shape_model_wrong_shape_4():
    model = np.array(
        [[1, 0]]
    )
    with pytest.raises(ValueError):
        shp = Shape(model)


def test_shape_number_of_points():
    shp = Shape(axon1(), n_points=100, refine=False)
    assert len(shp) == 100


def test_shape_number_of_points_None():
    shp = Shape(axon1(), n_points=None, refine=False)
    assert len(shp) == axon1().shape[1]


def test_shape_number_of_points_negative():
    with pytest.raises(ValueError):
        shp = Shape(axon1(), n_points=0)


def test_shape_number_of_points_float_number():
    with pytest.raises(ValueError):
        shp = Shape(axon1(), n_points=100.0)


def test_shape_number_of_points_string():
    with pytest.raises(ValueError):
        shp = Shape(axon1(), n_points='100')


def test_shape_number_of_points_noninteger():
    with pytest.raises(ValueError):
        shp = Shape(axon1(), n_points=(100,))


def test_shape_number_of_points_inf():
    with pytest.raises(ValueError):
        shp = Shape(axon1(), n_points=np.inf)


def test_shape_maximum_distance():
    shp = Shape(axon1(), dist_max=0.01)
    assert np.max(shp.dl) <= 0.01


def test_shape_maximum_distance_is_not_number_1():
    with pytest.raises(ValueError):
        shp = Shape(axon1(), dist_max='0.01')


def test_shape_maximum_distance_is_not_number_2():
    with pytest.raises(ValueError):
        shp = Shape(axon1(), dist_max=(0.01,))


def test_shape_maximum_distance_is_negative():
    with pytest.raises(ValueError):
        shp = Shape(axon1(), dist_max=-0.01)


def test_shape_maximum_distance_is_inf():
    with pytest.raises(ValueError):
        shp = Shape(axon1(), dist_max=np.inf)


def test_shape_x_shift():
    shp1 = Shape(axon1(), x_shift=-6)
    shp2 = Shape(axon1())
    shift = np.zeros_like(shp1.points)
    shift[0, :] = -6
    assert_almost_equal(shp1.points - shp2.points, shift)


def test_shape_y_shift():
    shp1 = Shape(axon1(), y_shift=3)
    shp2 = Shape(axon1())
    shift = np.zeros_like(shp1.points)
    shift[1, :] = 3
    assert_almost_equal(shp1.points - shp2.points, shift)


def test_shape_x_shift_not_numeric():
    with pytest.raises(ValueError):
        shp = Shape(axon1(), x_shift='10')


def test_shape_y_shift_not_numeric():
    with pytest.raises(ValueError):
        shp = Shape(axon1(), y_shift='10')


def test_shape_x_shift_inf():
    with pytest.raises(ValueError):
        shp = Shape(axon1(), x_shift=np.inf)


def test_shape_y_shift_not_numeric():
    with pytest.raises(ValueError):
        shp = Shape(axon1(), y_shift=np.inf)


def test_shape_rotate_angle():
    shp1 = Shape(axon1(), x_shift=5, y_shift=3, rotate_angle=40)
    shp2 = Shape(axon1(), x_shift=5, y_shift=3, rotate_angle=-320)
    assert_almost_equal(shp1.points, shp2.points)
    assert_almost_equal(shp1.normals, shp2.normals)


def test_shape_rotate_angle_not_numeric():
    with pytest.raises(ValueError):
        shp = Shape(axon1(), rotate_angle='90')


def test_shape_rotate_angle_is_inf():
    with pytest.raises(ValueError):
        shp = Shape(axon1(), rotate_angle=np.inf)


def test_shape_normals_with_shift_and_rotation():
    shp1 = Shape(axon1(), x_shift=3, y_shift=-10, rotate_angle=60)
    shp2 = Shape(axon1(), rotate_angle=60)
    assert_almost_equal(shp1.normals, shp2.normals)


def test_shape_normals_with_shift_and_rotation():
    shp1 = Shape(axon1(), x_shift=3, y_shift=-10, rotate_angle=60)
    shp2 = Shape(axon1(), rotate_angle=60)
    assert_almost_equal(shp1.normals, shp2.normals)


def test_shape_refine():
    shp1 = Shape(axon1(), refine=False)
    shp2 = Shape(axon1(), refine=True)
    assert len(shp1) < len(shp2)


def test_shape_plot():
    from matplotlib.lines import Line2D
    shp1 = Shape(axon1())
    h = shp1.plot()
    assert isinstance(h[0], Line2D)


def test_shape_repr_string():
    shp = Shape(axon1(), n_points=300, refine=False, rotate_angle=-60)
    desired_str = ('<Model2d>: n_points = 300,\n'
                   '           perimeter = 29.29,\n'
                   '           area = 50.98,\n'
                   '           rotate_angle = -60.')
    assert repr(shp) == desired_str


def test_shape_str_string():
    shp = Shape(axon1(), n_points=300, refine=False, rotate_angle=-60)
    desired_str = ('An imported model with\n'
                   '    perimeter = 29.29,\n'
                   '    area = 50.98,\n'
                   '    n_points = 300,\n'
                   '    rotated angle = -60 degree,\n'
                   '    max_curvature = 2.7905.')
    assert str(shp) == desired_str


# Test function project_stl.


def test_project_stl_input_random_path():
    with pytest.raises(ValueError):
        msh = project_stl('qwerty/uiop.stl')


def test_project_stl_path():
    from shapely.geometry.polygon import Polygon
    s = project_stl(mesh())
    assert isinstance(s, Polygon)


def test_project_stl_projection_best():
    s = project_stl(mesh(), projection='best')
    assert_almost_equal(s.area, 4.477129944465695)


def test_project_stl_projection_ndarray_1():
    normal = np.array([1, 0, 0])
    s = project_stl(mesh(), projection=normal)
    assert_almost_equal(s.area, 2.2008599129455018)


def test_project_stl_projection_ndarray_2():
    normal = np.array([[10], [0], [0]])
    s = project_stl(mesh(), projection=normal)
    assert_almost_equal(s.area, 2.2008599129455018)


def test_project_stl_projection_list_1():
    normal = [1, 0, 0]
    s = project_stl(mesh(), projection=normal)
    assert_almost_equal(s.area, 2.2008599129455018)


def test_project_stl_projection_list_2():
    normal = [1, 0, '0']
    s = project_stl(mesh(), projection=normal)
    assert_almost_equal(s.area, 2.2008599129455018)


def test_project_stl_projection_tuple_1():
    normal = (3, 0, 0)
    s = project_stl(mesh(), projection=normal)
    assert_almost_equal(s.area, 2.2008599129455018)


def test_project_stl_projection_tuple_2():
    normal = ('2', 0, 0)
    s = project_stl(mesh(), projection=normal)
    assert_almost_equal(s.area, 2.2008599129455018)


def test_project_stl_projection_random_string():
    with pytest.raises(ValueError):
        msh = project_stl(mesh(), projection='qwerty')


def test_project_stl_projection_wrong_type_1():
    with pytest.raises(ValueError):
        msh = project_stl(mesh(), projection=(1, 'a', 3))


def test_project_stl_projection_wrong_type_2():
    with pytest.raises(ValueError):
        msh = project_stl(mesh(), projection=dict())


def test_project_stl_projection_wrong_normal_vector():
    with pytest.raises(NameError):
        msh = project_stl(mesh(), projection=[0, 0, 0])


def test_project_stl_projection_negative_remesh_size():
    with pytest.raises(ValueError):
        msh = project_stl(mesh(), remesh_size=-0.1)
