"""Classes and functions for handling geometries."""

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.integrate import quad
import matplotlib.pyplot as plt
import alphashape
import trimesh
import shapely
from .utils import isnumeric, isint


class Ellipse:

    """
    Class of ellipses.

    If rotation angle is zero,
    the equation of the ellipse is:

    .. math::
        \\left( \\frac{x-center\_x}{a} \\right)^2 +
        \\left( \\frac{y-center\_y}{b} \\right)^2 = 1.

    Parameters
    ----------
    center_x : float
        x-coordinate of the ellipse center
    center_y : float
        y-coordinate of the ellipse center
    a : float
        length of the semi-major axis
    b : float
        length of the semi-minor axis
    n_points : int, optional
        number of points, by default 300
    dist_max : float or None, optional
        maximum length of edges in the discretized ellipse,
        by default None
    rotate_angle : float, optional
        rotation angle (degree), by default 0 degree
    refine : bool, optional
        if True, add more points on the high curvature region,
        by default True

    Attributes
    ----------
    points : (2, n_points) ndarray
        coordinates of sampled points
    normals : (2, n_points) ndarray
        outward pointing normal vectors of sampled points
    curvature : (n_points,) ndarray
        ellipse curvature
    dl : (n_points,) ndarray
        edge lengths
    """

    def __init__(self,
                 center_x,
                 center_y,
                 a,
                 b,
                 n_points=300,
                 dist_max=None,
                 rotate_angle=0,
                 refine=True) -> None:
        # input check
        if not isnumeric(center_x):
            raise ValueError(f"Ellipse: center_x={center_x} is not a number.")
        if not isnumeric(center_y):
            raise ValueError(f"Ellipse: center_y={center_y} is not a number.")
        if not isnumeric(a):
            raise ValueError(
                f"Ellipse: length of the semi-major axis, a={a}, is not a number.")
        if not isnumeric(b):
            raise ValueError(
                f"Ellipse: length of the semi-minor axis, b={b}, is not a number.")
        if not isint(n_points):
            raise ValueError(
                f"Ellipse: n_points={n_points} is not an integer.")
        if (dist_max is not None) and (not isnumeric(dist_max)):
            raise ValueError(f"Ellipse: dist_max={dist_max} is not a number.")
        if not isnumeric(rotate_angle):
            raise ValueError(
                f"Ellipse: rotate_angle={rotate_angle} is not a number.")
        if a <= 0:
            raise ValueError(
                f"Ellipse: length of the semi-major axis, a={a}, is negative or zero.")
        if b <= 0:
            raise ValueError(
                f"Ellipse: length of the semi-minor axis, b={b}, is negative or zero.")
        if n_points < 3:
            raise ValueError(f"Ellipse: n_points={n_points} is less than 3.")
        if (dist_max is not None) and (dist_max <= 0):
            raise ValueError(
                f"Ellipse: dist_max={dist_max} is negative or zero.")

        # get parameters
        self.center_x = center_x
        self.center_y = center_y
        self.a, self.b = a, b
        self.shift = np.array([[center_x], [center_y]])

        # get number of points
        if dist_max is None:
            self.n_points = n_points
        else:
            self.n_points = np.ceil(self.perimeter/dist_max).astype(int)

        # parameterize ellipse using ``_theta``
        self._theta = np.linspace(0, 1, self.n_points, endpoint=False)

        # get parameterized point cloud
        self.points = self._create_points()

        # compute curvature
        self.curvature = self._compute_curvature()

        # refine point cloud
        if refine:
            self._refine()

        # get out-pointing normal vectors
        self.normals = self._compute_normals()

        # compute length of edges
        self.dl = self._dl()

        # rotate the ellipse
        self.rotate_angle = rotate_angle
        if self.rotate_angle != 0:
            self._rotate()

        return None

    def plot(self):
        """Plot ellipse and normal vectors."""
        # plot ellipse
        h = plt.plot(self.points[0, :], self.points[1, :], 'b--')

        # plot normal vectors
        plt.quiver(self.points[0, :], self.points[1, :],
                   self.normals[0, :], self.normals[1, :])
        plt.axis('equal')

        return h

    @property
    def perimeter(self):
        """
        Ellipse perimeter.

        The perimeter is approximated using
        `Gauss-Kummer Series
        <https://mathworld.wolfram.com/Gauss-KummerSeries.html>`_.

        Returns
        -------
        float
            the 8th-order approximate perimeter
        """
        h = (self.a-self.b)**2/(self.a+self.b)**2

        return np.pi*(self.a+self.b)*(1 + h/4 + h**2/64 + h**3/256 +
                                      25*h**4/16384 + 49*h**5/65536 +
                                      441*h**6/1048576 + 1089*h**7/4194304)

    @property
    def area(self):
        """
        Ellipse area.

        Returns
        -------
        float
            ellipse area
        """
        return np.pi*self.a*self.b

    def _create_points(self):
        """
        Sample points on the ellipse.

        Returns
        -------
        (2, n_points) ndarray
            coordinates of points
        """
        x = self.a*np.cos(2*np.pi*self._theta)
        y = self.b*np.sin(2*np.pi*self._theta)

        return np.vstack((x, y)) + self.shift

    def _compute_curvature(self):
        """
        Compute ellipse curvature.

        According to
        `Wikipedia <https://en.wikipedia.org/wiki/Ellipse#Curvature>`_,
        the curvature is given by

        .. math::
            \\kappa = \\dfrac{ab}
            {\\left( \\frac{b^2x^2}{a^2}+\\frac{a^2y^2}{b^2} \\right)^
            {\\frac{3}{2}}}.

        Returns
        -------
        (n_points,) ndarray
            curvature
        """
        x = self.points[0, :]
        y = self.points[1, :]

        return (self.a*self.b)/((self.a*(y-self.center_y)/self.b)**2 +
                                (self.b*(x-self.center_x)/self.a)**2)**(3/2)

    def _compute_normals(self):
        """
        Compute outward pointing unit normal vectors
        on the ellipse boundary.

        Reference to the
        `answer <https://math.stackexchange.com/a/1673669/602901>`_.

        Returns
        -------
        (2, n_points) ndarray
            normal vectors
        """
        denom = np.array([self.a**2, self.b**2]).reshape((2, 1))
        points = self.points - self.shift
        normals = 2 * points / denom

        return normals/np.linalg.norm(normals, axis=0)

    def _dl(self):
        """
        Compute length of edges.

        The edges are centers on ``points``.

        Returns
        -------
        (n_points,) ndarray
            edge lengths
        """
        diff_pts = self.points - np.c_[self.points[:, 1:], self.points[:, 0:1]]
        dl = np.linalg.norm(diff_pts, axis=0)

        return ((dl + np.r_[dl[-1], dl[0:-1]])/2)

    def _rotate(self):
        """
        Rotate the ellipse and return the rotation matrix.

        Returns
        -------
        (2, 2) ndarray
            rotation matrix
        """
        # compute rotation matrix
        # https://en.wikipedia.org/wiki/Rotation_matrix#In_two_dimensions
        angle = self.rotate_angle*np.pi/180
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])

        self.points = R @ (self.points - self.shift) + self.shift
        self.normals = R @ self.normals

        return R

    def _refine(self) -> None:
        """
        Refine the disretization of the ellipse.

        Strategy of refinement: add more points to the region
        where the curvature is high.
        """
        # compute curvature distribution
        curva_abs = np.abs(self.curvature)
        curva_ratio = np.rint(
            np.sqrt(curva_abs*self.n_points / np.sum(curva_abs))).astype(int)

        # add points depending on the ratio of curvature
        n_points_refined = self.n_points + np.sum(curva_ratio)

        theta = np.zeros(n_points_refined)
        pointer = 0
        for i, v in enumerate(curva_ratio):
            # curvature ratio is small, keep the original points
            if v == 0:
                theta[pointer:pointer+v+1] = self._theta[i]
                pointer += v+1

            # special treatment to the first point
            elif i == 0:
                v_down = int(v/2)
                v_up = v - v_down
                if v_down > 0:
                    theta[-v_down:] = 1 + np.linspace((self._theta[-1]-1)/2,
                                                      0, v_down+1,
                                                      endpoint=False)[1:]
                theta[:v_up+1] = np.linspace(self._theta[0],
                                             (self._theta[1]+self._theta[0])/2,
                                             v_up+1)
                pointer += v_up+1

            # add points according to the curvature ratio
            else:
                theta[pointer:pointer+v+1] = (self._theta[i] -
                                              self._theta[i-1])/2 + \
                    np.linspace(self._theta[i],
                                self._theta[i-1], v+1, endpoint=False)[::-1]
                pointer += v+1

        # compute refined points
        self.n_points = n_points_refined
        self._theta = theta
        self.points = self._create_points()
        self.curvature = self._compute_curvature()

        return None

    def __len__(self):
        """Number of points."""
        return self.n_points

    def __repr__(self):
        """Ellipse information."""
        s = (f'<Ellipse>: center_x = {self.center_x},\n'
             f'           center_y = {self.center_y},\n'
             f'           a = {self.a}, b = {self.b},\n'
             f'           n_points = {self.n_points},\n'
             f'           rotate_angle = {self.rotate_angle}.')

        return s

    def __str__(self):
        """Ellipse information."""
        s = ('An ellipse with\n'
             f'    a = {self.a}, b = {self.b},\n'
             f'    centered at ({self.center_x}, {self.center_y}),\n'
             f'    rotated {self.rotate_angle} degrees\n'
             '    in anticlockwise direction.')

        return s


class Circle(Ellipse):

    """
    Class of circles.

    Parameters
    ----------
    center_x : float
        x-coordinate of the circle center
    center_y : float
        y-coordinate of the circle center
    r : float
        circle radius
    n_points : int, optional
        number of points, by default 300
    dist_max : float, optional
        maximum length of edges in the discretized circle,
        by default None
    """

    def __init__(self,
                 center_x,
                 center_y,
                 r,
                 n_points=300,
                 dist_max=None) -> None:
        # input check
        if not isnumeric(center_x):
            raise ValueError(f"Circle: center_x={center_x} is not a number.")
        if not isnumeric(center_y):
            raise ValueError(f"Circle: center_y={center_y} is not a number.")
        if not isnumeric(r):
            raise ValueError(f"Circle: r={r} is not a number.")
        if not isint(n_points):
            raise ValueError(f"Circle: n_points={n_points} is not an integer.")
        if (dist_max is not None) and (not isnumeric(dist_max)):
            raise ValueError(f"Circle: dist_max={dist_max} is not a number.")
        if r <= 0:
            raise ValueError(f"Circle: r={r} is negative or zero.")
        if n_points < 3:
            raise ValueError(f"Circle: n_points={n_points} is less than 3.")
        if (dist_max is not None) and (dist_max <= 0):
            raise ValueError(
                f"Circle: dist_max={dist_max} is negative or zero.")

        # initialize parent class
        super().__init__(center_x,
                         center_y,
                         r,
                         r,
                         n_points=n_points,
                         dist_max=dist_max,
                         refine=False)
        self.r = r
        self.curvature = np.ones_like(self.curvature)/r

        return None

    def __repr__(self):
        """Circle information."""
        s = (f'<Circle(Ellipse)>: center_x = {self.center_x},\n'
             f'                   center_y = {self.center_y},\n'
             f'                   r = {self.a},\n'
             f'                   n_points = {self.n_points}.')

        return s

    def __str__(self):
        """Circle information."""
        s = ('A circle with\n'
             f'    radius = {self.r},\n'
             f'    centered at ({self.center_x}, {self.center_y}).')

        return s


class Shape:

    """
    Class for 2D shapes.

    Parameters
    ----------
    model : ``shapely.geometry.Polygon`` or ``numpy.ndarray``
        input model as a polygon or a point cloud
    n_points : int or None, optional
        number of points, by default None
    dist_max : float or None, optional
        maximum length of edges in the discretized contour,
        by default None
    x_shift : float, optional
        translation distance on the x-axis, by default 0
    y_shift : float, optional
        translation distance on the y-axis, by default 0
    rotate_angle : float, optional
        rotation angle (degree), by default 0 degree
    refine : bool, optional
        if True, add more points on the high curvature region,
        by default True

    Raises
    ------
    ValueError
        dimension of the point cloud is wrong
    NotImplementedError
        input model not implemented

    Attributes
    ----------
    points : (2, n_points) ndarray
        coordinates of sampled points
    normals : (2, n_points) ndarray
        outward pointing normal vectors of sampled points
    curvature : (n_points,) ndarray
        curvature on shape's contour
    dl : (n_points,) ndarray
        edge lengths
    """

    def __init__(self,
                 model,
                 n_points=None,
                 dist_max=None,
                 x_shift=0,
                 y_shift=0,
                 rotate_angle=0,
                 refine=True):
        # input check
        if n_points is not None:
            if not isint(n_points):
                raise ValueError(f"Shape: n_points={n_points} is not integer.")
            elif n_points <= 0:
                raise ValueError(
                    f"Shape: n_points={n_points} is negative or zero.")

        if dist_max is not None:
            if not isnumeric(dist_max):
                raise ValueError(
                    f"Shape: dist_max={dist_max} is not a number.")
            elif dist_max <= 0:
                raise ValueError(f"Shape: dist_max={dist_max} is negative.")

        if not isnumeric(x_shift):
            raise ValueError(f"Shape: x_shift={x_shift} is not a number.")

        if not isnumeric(y_shift):
            raise ValueError(f"Shape: y_shift={y_shift} is not a number.")

        if not isnumeric(rotate_angle):
            raise ValueError(
                f"Shape: rotate_angle={rotate_angle} is not a number.")

        # get point clouds
        # model type: Polygon
        if isinstance(model, shapely.geometry.Polygon):
            # extract point coordinates
            points = np.array(list(model.exterior.coords)).T
            # enforce counterclockwise order
            if is_clockwise(points):
                points = points[:, ::-1]
            # B-cubic-spline interpolation
            self._tck, self._u = splprep(points, u=None, s=0.0, per=1)

        # model type: point cloud
        elif isinstance(model, np.ndarray):
            points = model
            if len(points.shape) == 2 and \
                    points.shape[0] == 2 and \
                    points.shape[1] >= 3:
                # enforce counterclockwise order
                if is_clockwise(points):
                    points = points[:, ::-1]
                # B-cubic-spline interpolation
                self._tck, self._u = splprep(points, u=None, s=0.0, per=1)
            else:
                error_msg = "".join((
                    "Input model should be a 2xN (N>=3) ndarray ",
                    "containing the coordinates of sampling points ",
                    "on a closed 2D curve."
                ))
                raise ValueError(error_msg)

        # other model type
        else:
            raise NotImplementedError('Input model type is not implemented.')

        # get number of points
        if n_points is not None:
            self.n_points = n_points
        elif dist_max is not None:
            self.n_points = np.ceil(self.perimeter/dist_max).astype(int)
        else:
            self.n_points = points.shape[1]

        # get shift
        self.x_shift = x_shift
        self.y_shift = y_shift
        self.shift = np.array([[x_shift], [y_shift]])

        # parameterize shape contour using ``u``
        self.u = np.linspace(
            self._u.min(), self._u.max(), self.n_points, endpoint=False)

        # compute curvature
        self.curvature = self._compute_curvature()

        # refine point cloud
        if refine:
            self._refine()

        # recompute points and normal vectors
        self.points = self._create_points()
        self.normals = self._compute_normals()

        # compute length of edges
        self.dl = self._dl()

        # rotate the 2D shape
        self.rotate_angle = rotate_angle
        if self.rotate_angle != 0:
            self._rotate()

        return None

    def plot(self):
        """Plot the 2D shape and normal vectors."""
        # plot shape
        h = plt.plot(self.points[0, :], self.points[1, :], 'b--')

        # plot normal vectors
        plt.quiver(self.points[0, :], self.points[1, :],
                   self.normals[0, :], self.normals[1, :])
        plt.axis('equal')

        return h

    @property
    def perimeter(self):
        """
        Perimeter of shape contour.

        The perimeter is calculated by `integrating the arc lengths`_.

        .. _integrating the arc lengths:
            https://en.wikipedia.org/wiki/Arc_length#Finding_arc_lengths_by_integrating
        """
        # compute arc length
        def len_func(u, tck):
            x_prime, y_prime = splev(u, tck, der=1)
            return np.sqrt(y_prime**2 + x_prime**2)

        # integrate the arc lengths
        l, _ = quad(len_func, 0, 1, args=(
            self._tck,), limit=2000)

        return l

    @property
    def area(self):
        """
        Shape area.

        Shape area is calculated by using the `Green's theorem`_.

        .. _Green's theorem:
            https://mathworld.wolfram.com/GreensTheorem.html#eqn4
        """
        # define integrand
        def area_func(u, tck):
            x, y = splev(u, tck, der=0)
            x_prime, y_prime = splev(u, tck, der=1)
            return (x*y_prime - y*x_prime)/2

        # compute area
        a, _ = quad(area_func, 0, 1, args=(
            self._tck,), limit=2000)

        return a

    def _create_points(self):
        """
        Sample points on the shape contour.

        Returns
        -------
        (2, n_points) ndarray
            coordinates of points
        """
        x, y = splev(self.u, self._tck, der=0)

        return np.vstack((x, y)) + self.shift

    def _compute_curvature(self):
        """
        Compute curvature of a arbitrary 2D contour.

        According to
        `this`_,
        the curvature is given by

        .. math::
            \\kappa = \\dfrac{x' y'' - y'x''}
            {\\left(x'^2 + y'^2\\right)^{\\frac{3}{2}}}.

        .. _this:
            https://en.wikipedia.org/wiki/Curvature#In_terms_of_a_general_parametrization

        Returns
        -------
        (n_points,) ndarray
            curvature
        """
        x_deri1, y_deri1 = splev(self.u, self._tck, der=1)
        x_deri2, y_deri2 = splev(self.u, self._tck, der=2)

        return (x_deri1*y_deri2-x_deri2*y_deri1)/(x_deri1**2+y_deri1**2)**(3/2)

    def _compute_normals(self):
        """
        Compute outward pointing unit normal vectors on the shape contour.

        The normal vectors are perpendicular to the tangent vectors.

        Returns
        -------
        (2, n_points) ndarray
            normal vectors
        """
        x_deri1, y_deri1 = splev(self.u, self._tck, der=1)
        normals = np.vstack((y_deri1, -x_deri1))

        return normals/np.linalg.norm(normals, axis=0)

    def _dl(self):
        """
        Compute length of edges.

        The edges are centers on ``points``.

        Returns
        -------
        (n_points,) ndarray
            edge lengths
        """
        diff_pts = self.points - np.c_[self.points[:, 1:], self.points[:, 0:1]]
        dl = np.linalg.norm(diff_pts, axis=0)
        return (dl + np.r_[dl[-1], dl[0:-1]])/2

    def _rotate(self):
        """
        Rotate the shape and return the rotation matrix.

        Returns
        -------
        (2, 2) ndarray
            rotation matrix
        """
        # compute rotation matrix
        # https://en.wikipedia.org/wiki/Rotation_matrix#In_two_dimensions
        angle = self.rotate_angle*np.pi/180
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])

        self.points = R @ (self.points - self.shift) + self.shift
        self.normals = R @ self.normals

        return R

    def _refine(self) -> None:
        """
        Refine the disretization of the ellipse.

        Strategy of refinement: add more points to the region
        whose curvature is high.
        """
        # compute curvature distribution
        curva_abs = np.abs(self.curvature)
        curva_ratio = np.rint(
            np.sqrt(curva_abs*self.n_points / np.sum(curva_abs))).astype(int)

        # add points depending on the ratio of curvature
        n_points_refined = self.n_points + np.sum(curva_ratio)

        u = np.zeros(n_points_refined)
        pointer = 0
        for i, v in enumerate(curva_ratio):
            # curvature ratio is small, keep the original points
            if v == 0:
                u[pointer:pointer+v+1] = self.u[i]
                pointer += v+1

            # special treatment to the first point
            elif i == 0:
                v_down = int(v/2)
                v_up = v - v_down
                if v_down > 0:
                    u[-v_down:] = 1 + np.linspace((self.u[-1]-1)/2,
                                                  0, v_down+1,
                                                  endpoint=False)[1:]
                u[:v_up+1] = np.linspace(self.u[0],
                                         (self.u[1]+self.u[0])/2, v_up+1)
                pointer += v_up+1

            # add points according to the curvature ratio
            else:
                u[pointer:pointer+v+1] = (self.u[i]-self.u[i-1])/2 + \
                    np.linspace(self.u[i], self.u[i-1],
                                v+1, endpoint=False)[::-1]
                pointer += v+1

        # compute refined points
        self.n_points = n_points_refined
        self.u = u
        self.curvature = self._compute_curvature()

        return None

    def __len__(self):
        """Number of points."""
        return self.n_points

    def __repr__(self):
        """2D model information."""
        s = (f'<Model2d>: n_points = {self.n_points},\n'
             f'           perimeter = {self.perimeter:.2f},\n'
             f'           area = {self.area:.2f},\n'
             f'           rotate_angle = {self.rotate_angle}.')

        return s

    def __str__(self):
        """2D model information."""
        s = ('An imported model with\n'
             f'    perimeter = {self.perimeter:.2f},\n'
             f'    area = {self.area:.2f},\n'
             f'    n_points = {self.n_points},\n'
             f'    rotated angle = {self.rotate_angle} degree,\n'
             f'    max_curvature = {np.max(self.curvature):.4f}.')

        return s


def project_stl(stl_path, projection='best', remesh_size=1, alpha=1.5):
    """
    Project a 3D STL surface mesh onto a plane to form a 2D domain.

    Parameters
    ----------
    stl_path : string
        path to a STL file
    projection : (3,) ndarray, optional
        unit normal vector of a projection plane, by default 'best'
    remesh_size : float, optional
        maximum length of any edge in the result, by default 1
    alpha : float, optional
        alphashape's alpha value, by default 1.5

    Returns
    -------
    ``shapely.geometry.Polygon``
        the resulting geometry
    """
    # load 3D model
    mymesh = trimesh.load_mesh(stl_path)

    # refine model (mandatory)
    if remesh_size <= 0:
        raise ValueError("'remesh_size' is negative.")
    vertices, _ = trimesh.remesh.subdivide_to_size(
        mymesh.vertices, mymesh.faces, remesh_size)

    # get projection direction
    if isinstance(projection, (list, tuple, np.ndarray)) and len(projection) == 3:
        projection = np.array(projection, dtype=float).reshape(-1)
        center = vertices.mean(axis=0)
    elif isinstance(projection, str) and projection == 'best':
        center, projection = trimesh.points.plane_fit(vertices)
    else:
        raise ValueError(f"Unknown parameter 'projection'={projection}.")

    # project 3D model to the plane
    points = trimesh.points.project_to_plane(
        vertices-center, plane_normal=projection)

    # return the boundary
    return alphashape.alphashape(points, alpha)


def is_clockwise(P1):
    """
    Determine the winding order of a set of ordered points ``P1``.

    Reference to the `blog`_ and the `post`_.

    .. _blog:
        https://www.element84.com/blog/determining-the-winding-of-a-polygon-given-as-a-set-of-ordered-points

    .. _post:
        https://stackoverflow.com/a/1165943/8933288

    Parameters
    ----------
    P1 : (2, n) ndarray
        a list of ordered points from a polygon

    Returns
    -------
    bool
        True, if P1 is clockwisely ordered
    """
    P2 = np.c_[P1[:, 1:], P1[:, 0:1]]

    return np.sum((P2[0, :] - P1[0, :])*(P2[1, :] + P1[1, :])) > 0
