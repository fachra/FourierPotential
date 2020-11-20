import numpy as np
from scipy.interpolate import splprep, splev
from scipy.integrate import quad
import matplotlib.pyplot as plt
import alphashape
import trimesh
import shapely


def load_stl(stl_path, projection='best', remesh_size=1, alpha=1.5):
    # load 3D model
    mymesh = trimesh.load_mesh(stl_path)
    # refine model (mandatory)
    vertices, _ = trimesh.remesh.subdivide_to_size(
        mymesh.vertices, mymesh.faces, remesh_size)
    # projection direction
    if projection == 'best':
        center, projection = trimesh.points.plane_fit(vertices)
    else:
        center = vertices.mean(axis=0)
    # 3D model to 2D model
    points = trimesh.points.project_to_plane(
        vertices-center, plane_normal=projection)
    # return the boundary
    return alphashape.alphashape(points, alpha)


class Ellipse:
    '''
    Class of ellipses.
        The equation of the ellipse is:
        ((x-center_x)/a)**2 + ((y-center_y)/b)**2 = 1
    '''

    def __init__(self, center_x, center_y, a, b, n_points=300,
                 dist_max=None, rotate_angle=0, refine=True) -> None:
        self.center_x = center_x
        self.center_y = center_y
        self.a, self.b = a, b
        self.area = np.pi*a*b
        self.perimeter = self._perimeter()

        if dist_max is None:
            self.n_points = n_points
        else:
            self.n_points = np.ceil(self.perimeter/dist_max).astype(int)

        self.theta = np.linspace(0, 1, self.n_points, endpoint=False)
        self.points = self._points()
        self.curvature = self._curvature()
        if refine:
            self.refine()
        self.normals = self._normals()
        self.dl = self._dl()
        if rotate_angle != 0:
            self._rotate(rotate_angle)

    def _points(self):
        x = self.a*np.cos(2*np.pi*self.theta) + self.center_x
        y = self.b*np.sin(2*np.pi*self.theta) + self.center_y
        return np.vstack((x, y))

    def _perimeter(self):
        """
        Return the approximate perimeter of an ellipse given by Gauss-Kummer Series.
        https://mathworld.wolfram.com/Gauss-KummerSeries.html

        Parameters
        ----------
        a : float
            The length of semi-major axis.
        b : float
            The length of semi-minor axis.

        Returns
        -------
        perimeter : float
            The 8th order approximate perimeter.
        """
        h = (self.a-self.b)**2/(self.a+self.b)**2
        return np.pi*(self.a+self.b)*(1+h/4+h**2/64+h**3/256+25 *
                                      h**4/16384+49*h**5/65536+441*h**6/1048576+1089*h**7/4194304)

    def _curvature(self):
        '''
        https://en.wikipedia.org/wiki/Ellipse#Curvature
        '''
        x = self.points[0, :]
        y = self.points[1, :]
        return (self.a*self.b)/((self.a*(y-self.center_y)/self.b)**2 +
                                (self.b*(x-self.center_x)/self.a)**2)**(3/2)

    def _normals(self):
        x_deri1 = -2*np.pi*self.a*np.sin(2*np.pi*self.theta)
        y_deri1 = 2*np.pi*self.b*np.cos(2*np.pi*self.theta)
        normals = np.vstack((y_deri1, -x_deri1))
        return normals/np.linalg.norm(normals, axis=0)

    def _dl(self):
        dl = np.linalg.norm(
            self.points - np.c_[self.points[:, 1:], self.points[:, 0:1]], axis=0)
        return ((dl + np.r_[dl[-1], dl[0:-1]])/2)

    def _rotate(self, angle):
        '''
        https://en.wikipedia.org/wiki/Rotation_matrix#In_two_dimensions
        '''
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        self.points = np.matmul(R, self.points)
        self.normals = np.matmul(R, self.normals)
        return R

    def refine(self):
        '''
        Strategy of refinement: add more points to the region whose curvature is large.
        '''
        curva_abs = np.abs(self.curvature)
        curva_ratio = np.rint(
            np.sqrt(curva_abs*self.n_points / np.sum(curva_abs))).astype(int)
        # the addition of points depends on the ratio of curvature
        n_points_refined = self.n_points + np.sum(curva_ratio)

        theta = np.zeros(n_points_refined)
        pointer = 0
        for i, v in enumerate(curva_ratio):
            if v == 0:
                # curvature ratio is small, keep the original points
                theta[pointer:pointer+v+1] = self.theta[i]
                pointer += v+1
            elif i == 0:
                # special treatment to the first point
                v_down = int(v/2)
                v_up = v - v_down
                if v_down > 0:
                    theta[-v_down:] = np.linspace((self.theta[-1]-1)/2,
                                                  0, v_down+1, endpoint=False)[1:] + 1
                theta[:v_up+1] = np.linspace(self.theta[0],
                                             (self.theta[1]+self.theta[0])/2, v_up+1)
                pointer += v_up+1
            else:
                # add points according to the curvature ratio
                theta[pointer:pointer+v+1] = \
                    np.linspace(self.theta[i], self.theta[i-1], v+1,
                                endpoint=False)[::-1] + (self.theta[i]-self.theta[i-1])/2
                pointer += v+1

        # compute refined points
        self.n_points = n_points_refined
        self.theta = theta
        self.points = self._points()
        self.curvature = self._curvature()
        return 0

    def plot(self):
        fig, axs = plt.subplots(1)
        axs.plot(self.points[0, :], self.points[1, :], 'b--')
        axs.quiver(self.points[0, :], self.points[1, :],
                   self.normals[0, :], self.normals[1, :])
        axs.axis('equal')
        return fig


class Circle(Ellipse):
    def __init__(self, center_x, center_y, r, n_points=300, dist_max=None) -> None:
        Ellipse.__init__(self, center_x, center_y, r, r,
                         n_points=n_points, dist_max=dist_max, refine=False)
        self.perimeter = 2*np.pi*r
        self.curvature = np.ones_like(self.curvature)/r


class Model2D:
    '''
    Class of external models.
    '''

    def __init__(self, model, n_points=None, dist_max=None,
                 x_shift=0, y_shift=0, rotate_angle=0, refine=True) -> None:
        # model type: Polygon
        if isinstance(model, shapely.geometry.polygon.Polygon):
            points = np.array(list(model.exterior.coords)).T
            if self._is_clockwise(points):
                # enforce counterclockwise order
                points = points[:, ::-1]
            # B-cubic-spline interpolation
            self._tck, self._u = splprep(points, u=None, s=0.0, per=1)
            self.area = model.area
            self.perimeter = model.length

        # model type: point cloud
        if isinstance(model, np.ndarray):
            points = model
            if len(points.shape) == 2 and points.shape[0] == 2 and points.shape[1] >= 3:
                if self._is_clockwise(points):
                    points = points[:, ::-1]
                self._tck, self._u = splprep(points, u=None, s=0.0, per=1)
                self.area = self._area()
                self.perimeter = self._perimeter()
            else:
                raise ValueError(
                    'Input data should be a 2xN (N>=3) array containing the coordinates of sampling points on a closed 2D curve.')

        if n_points is not None:
            self.n_points = n_points
        elif dist_max is not None:
            self.n_points = np.ceil(self.perimeter/dist_max).astype(int)
        else:
            self.n_points = points.shape[1]

        self.u = np.linspace(
            self._u.min(), self._u.max(), self.n_points, endpoint=False)
        self.curvature = self._curvature()

        if refine:
            self.refine()

        self.points = self._points(x_shift, y_shift)
        self.normals = self._normals()
        self.dl = self._dl()
        if rotate_angle != 0:
            self._rotate(rotate_angle)

    def _is_clockwise(self, P1):
        '''
        https://www.element84.com/blog/determining-the-winding-of-a-polygon-given-as-a-set-of-ordered-points
        https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
        '''
        P2 = np.c_[P1[:, 1:], P1[:, 0:1]]
        return np.sum((P2[0, :] - P1[0, :])*(P2[1, :] + P1[1, :])) > 0

    def _area(self):
        '''
        https://mathworld.wolfram.com/GreensTheorem.html
        '''
        def area_func(u, tck):
            x, y = splev(u, tck, der=0)
            x_prime, y_prime = splev(u, tck, der=1)
            return (x*y_prime - y*x_prime)/2
        a, _ = quad(area_func, 0, 1, args=(
            self._tck,), limit=2000)
        return a

    def _perimeter(self):
        '''
        https://en.wikipedia.org/wiki/Arc_length#Finding_arc_lengths_by_integrating
        '''
        def len_func(u, tck):
            x_prime, y_prime = splev(u, tck, der=1)
            return np.sqrt(y_prime**2 + x_prime**2)
        l, _ = quad(len_func, 0, 1, args=(
            self._tck,), limit=2000)
        return l

    def _curvature(self):
        x_deri1, y_deri1 = splev(self.u, self._tck, der=1)
        x_deri2, y_deri2 = splev(self.u, self._tck, der=2)
        return (x_deri1*y_deri2-x_deri2*y_deri1)/(x_deri1**2+y_deri1**2)**(3/2)

    def _points(self, x_shift, y_shift):
        x, y = splev(self.u, self._tck, der=0)
        return np.vstack((x+x_shift, y+y_shift))

    def _normals(self):
        x_deri1, y_deri1 = splev(self.u, self._tck, der=1)
        normals = np.vstack((y_deri1, -x_deri1))
        return normals/np.linalg.norm(normals, axis=0)

    def _dl(self):
        dl = np.linalg.norm(
            self.points - np.c_[self.points[:, 1:], self.points[:, 0:1]], axis=0)
        return (dl + np.r_[dl[-1], dl[0:-1]])/2

    def _rotate(self, angle):
        '''
        https://en.wikipedia.org/wiki/Rotation_matrix#In_two_dimensions
        '''
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        self.points = np.matmul(R, self.points)
        self.normals = np.matmul(R, self.normals)
        return R

    def refine(self):
        '''
        Strategy of refinement: add more points to the region whose curvature is large.
        '''
        curva_abs = np.abs(self.curvature)
        curva_ratio = np.rint(
            np.sqrt(curva_abs*self.n_points / np.sum(curva_abs))).astype(int)
        n_points_refined = self.n_points + np.sum(curva_ratio)

        u = np.zeros(n_points_refined)
        pointer = 0
        for i, v in enumerate(curva_ratio):
            if v == 0:
                u[pointer:pointer+v+1] = self.u[i]
                pointer += v+1
            elif i == 0:
                v_down = int(v/2)
                v_up = v - v_down
                if v_down > 0:
                    u[-v_down:] = np.linspace((self.u[-1]-1)/2,
                                              0, v_down+1, endpoint=False)[1:] + 1
                u[:v_up+1] = np.linspace(self.u[0],
                                         (self.u[1]+self.u[0])/2, v_up+1)
                pointer += v_up+1
            else:
                u[pointer:pointer+v+1] = \
                    np.linspace(self.u[i], self.u[i-1], v+1,
                                endpoint=False)[::-1] + (self.u[i]-self.u[i-1])/2
                pointer += v+1

        self.n_points = n_points_refined
        self.u = u
        self.curvature = self._curvature()
        return 0

    def plot(self):
        fig, axs = plt.subplots(1)
        axs.plot(self.points[0, :], self.points[1, :], 'b--')
        axs.quiver(self.points[0, :], self.points[1, :],
                   self.normals[0, :], self.normals[1, :])
        axs.axis('equal')
        return fig
