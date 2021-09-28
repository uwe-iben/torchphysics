import warnings
import numpy as np
import numbers
from .domain import Domain, LambdaDomain


class Point(Domain):
    """Creates a single point at the given coordinates.

    Parameters
    ----------
    space : Space
        The space in which this object lays.
    coord : Number, List or callable
        The coordinate of the point.
    """
    def __new__(cls, space, coord, tol=1e-06):
        if callable(coord):
            return LambdaDomain(Point, {'coord': coord},
                                space=space, dim=0, tol=tol)
        return super(Point, cls).__new__(cls)

    def __init__(self, space, coord, tol=1e-06):
        super().__init__(space, dim=0, tol=tol)
        if isinstance(coord, numbers.Number):
            coord = [coord]
        self.coord = coord

    def is_inside(self, points):
        points = super()._check_single_point(points)
        points = super().return_space_variables_to_point_list(points)
        inside = (np.linalg.norm(points - self.coord, axis=1) <= self.tol)
        return inside.reshape(-1, 1)

    def sample_random_uniform(self, n):
        points = self.coord * np.ones((n, 1))
        return super().divide_points_to_space_variables(points)

    def sample_grid(self, n):
        # for one single point grid and random sampling is the same
        return self.sample_random_uniform(n)

    def __add__(self, other):
        assert other.dim == 0
        if isinstance(other, PointCloud):
            return other + self
        if all(np.isclose(other.coord, self.coord, atol=self.tol)):
            return self
        else:
            new_coords = np.stack((self.coord, other.coord))
            return PointCloud(space=self.space, coord_list=new_coords, 
                              tol=np.min([self.tol, other.tol]))

    def __sub__(self, other):
        assert other.dim == 0
        if isinstance(other, PointCloud):
            if any(other._check_coords_inside([self.coord], other.coord_list)):
                warnings.warn("""Cut is empty""")
                return 
        elif all(np.isclose(other.coord, self.coord, atol=self.tol)):
            warnings.warn("""Cut is empty""")
            return 
        return self

    def __and__(self, other):
        assert other.dim == 0
        if isinstance(other, PointCloud):
            if any(other._check_coords_inside([self.coord], other.coord_list)):
                return self
        elif all(np.isclose(other.coord, self.coord, atol=self.tol)):
            return self
        warnings.warn("""Intersection is empty""")
        return


class PointCloud(Domain):
    """Creates a point cloud of many single points.

    Parameters
    ----------
    space : Space
        The space in which this object lays.
    coord_list : List of lists or callables
        The coordinates of all the point. Each entry of the list will
        be its own point.  
    """
    def __new__(cls, space, coord_list, tol=1e-06):
        if callable(coord_list):
            params = {'coord_list': coord_list}
            return LambdaDomain(class_=cls, params=params, space=space,
                                dim=2, tol=tol) 
        return super(PointCloud, cls).__new__(cls)

    def __init__(self, space, coord_list, tol=1e-06):
        super().__init__(space, dim=0, tol=tol)
        self.coord_list = np.array(coord_list).astype(np.float32)

    def is_inside(self, points):
        points = super()._check_single_point(points)
        points = super().return_space_variables_to_point_list(points)
        inside = np.zeros((len(points), 1), dtype=bool)
        for i in range(len(points)):
            dist = np.linalg.norm(self.coord_list - points[i], axis=1)
            inside[i] = any(dist <= self.tol) 
        return super().is_inside(points)

    def bounding_box(self):
        point_dim = len(self.coord_list[0])
        bounds = []
        for i in range(point_dim):
            min_ = np.min(self.coord_list[:, i])
            max_ = np.max(self.coord_list[:, i])
            bounds.append(min_)
            bounds.append(max_)
        return bounds

    def sample_random_uniform(self, n):
        index = np.random.choice(len(self.coord_list), n, replace=True)
        return super().divide_points_to_space_variables(self.coord_list[index])

    def sample_grid(self, n):
        len_of_coord_list = len(self.coord_list)
        index = np.tile(range(len_of_coord_list), int(np.ceil(n/len_of_coord_list)))
        return super().divide_points_to_space_variables(self.coord_list[index[:n]])

    def __add__(self, other):
        assert other.dim == 0
        other_coord_list = self._get_other_coord_list(other)
        inside = self._check_coords_inside(other_coord_list, self.coord_list)
        index = np.where(np.invert(inside))[0]
        self.coord_list = np.append(self.coord_list, 
                                    other_coord_list[index], 
                                    axis=0)
        return self._new_point_object()

    def __sub__(self, other):
        assert other.dim == 0
        other_coord_list = self._get_other_coord_list(other)
        inside = self._check_coords_inside(self.coord_list, other_coord_list)
        index = np.where(np.invert(inside))[0]
        self.coord_list = self.coord_list[index]
        return self._new_point_object()

    def __and__(self, other):
        assert other.dim == 0
        other_coord_list = self._get_other_coord_list(other)
        inside = self._check_coords_inside(self.coord_list, other_coord_list)
        index = np.where(inside)[0]
        self.coord_list = self.coord_list[index]
        return self._new_point_object()

    def _get_other_coord_list(self, other):
        if isinstance(other, Point):
            return np.array([other.coord])
        return other.coord_list

    def _check_coords_inside(self, list_1, list_2):
        inside = np.zeros((len(list_1), 1), dtype=bool)
        for i in range(len(list_1)):
            for coord in list_2:
                if np.allclose(list_1[i], coord, atol=self.tol):
                    inside[i] = True
                    break
        return inside

    def _new_point_object(self):
        if len(self.coord_list) == 0:
            warnings.warn("""The new PointCloud is empty""")
            return
        elif len(self.coord_list) == 1:
            return Point(space=self.space, coord=self.coord_list[0], atol=self.tol)
        return self