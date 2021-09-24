import abc
import numpy as np


class Domain:
    def __init__(self, space, dim=None, tol=1e-06):
        self.space = space
        self.tol = tol
        if dim is None:
            self.dim = self.space.dim
        else:
            self.dim = dim

    @property
    def is_initialized(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __add__(self, other):
        """Creates the union of the two input domains.

        Parameters
        ----------
        other : Domain
            The other domain that should be united with the domain.
            Has to be of the same dimension.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __sub__(self, other):
        """Creates the cut of the two input domains.

        Parameters
        ----------
        other : Domain
            The other domain that should be cut off the domain.
            Has to be of the same dimension.
        """
        raise NotImplementedError

    @abc.abstractmethod    
    def __and__(self, other):
        """Creates the intersection of the two input domains.

        Parameters
        ----------
        other : Domain
            The other domain that should be intersected with the domain.
            Has to be of the same dimension.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_inside(self, points):
        raise NotImplementedError

    @abc.abstractmethod
    def bounding_box(self):
        raise NotImplementedError

    @abc.abstractmethod
    def sample_grid(self, n):
        raise NotImplementedError

    @abc.abstractmethod
    def sample_random_uniform(self, n):
        raise NotImplementedError

    def __mul__(self, other):
        return ProductDomain(self, other)

    @property
    def boundary(self):
        # Domain object of the boundary
        raise NotImplementedError

    @property
    def inner(self):
        # open domain
        raise NotImplementedError

    def _cut_points(self, n, points):
        """Deletes some random points, if more than n were sampled
        (can for example happen by grid-sampling).
        """
        if len(points) > n:
            index = np.random.choice(len(points), int(n), replace=False)
            return points[index]
        return points

    def _check_grid_enough_points(self, n, points):
        """Checks if there are not enough points for the grid.
        If not, add some random points.
        """ 
        if len(points) < n:
            new_points = self.sample_random_uniform(n-len(points))
            points = np.append(points, new_points, axis=0)
        return points

    def _check_single_point(self, points):
        if len(np.shape(points)) == 1:
            points = np.array([points])
        return points


class LambdaDomain(Domain):
    def __init__(self, class_, params, space, dim, tol):
        super().__init__(space, dim=dim, tol=tol)
        self.class_ = class_
        self.params = params

    def __call__(self, point):
        p = {}
        for k in self.params:
            if callable(self.params[k]):
                p[k] = self.params[k](point)
            else:
                p[k] = self.params[k]
        return self.class_(space=self.space, tol=self.tol, **p)
    
    def __mul__(self, other):
        return super().__mul__(other)


class BoundaryDomain(Domain):
    def __init__(self, domain):
        super().__init__(domain.space, dim=domain.dim-1, tol=domain.tol)

    @abc.abstractmethod
    def normal(self, points):
        pass