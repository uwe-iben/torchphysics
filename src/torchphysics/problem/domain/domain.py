import abc
import numpy as np
import warnings


class Domain:
    def __init__(self, space, dim=None, tol=1e-06):
        self.space = space
        self.tol = tol
        if dim is None:
            self.dim = self.space.dim
        else:
            self.dim = dim

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
        """Creates the cut of domain other from self.

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

    @abc.abstractmethod
    def project_on_subspace(self, subspace):
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


class BoundaryDomain(Domain):
    def __init__(self, domain):
        super().__init__(domain.space, dim=domain.dim-1, tol=domain.tol)

    @abc.abstractmethod
    def normal(self, points):
        pass


class ProductDomain(Domain):
    def __init__(self, domain_a, domain_b):
        self.domain_a = domain_a
        self.domain_b = domain_b
        if not self.domain_a.space.keys().isdisjoint(self.domain_b.space):
            warnings.warn("""Warning: The space of a ProductDomain will be the product
                of its factor domains spaces. This may lead to unexpected behaviour.""")
        space = self.domain_a.space * self.domain_b.space
        super().__init__(space=space,
                         dim=domain_a.dim + domain_b.dim,
                         tol=min(domain_a.tol, domain_b.tol))

    def project_on_subspace(self, subspace=None, **variables):
        assert subspace in self.space
        for domain in (self.domain_a, self.domain_b):
            if subspace in domain.space:
                return domain.project_on_subspace(subspace)
        return ProductDomain(self.domain_a.project_on_subspace(subspace & self.domain_a.space),
                             self.domain_b.project_on_subspace(subspace & self.domain_b.space))
        # TODO: fix this to keep order of dimensions with identical names
        

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
        """Creates the cut of other from self.

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
        return ProductDomain(self.domain_a & other, self.domain_b & other)

    @abc.abstractmethod
    def is_inside(self, points):
        return 

    @abc.abstractmethod
    def bounding_box(self):
        raise NotImplementedError

    @abc.abstractmethod
    def sample_grid(self, n):
        raise NotImplementedError

    @abc.abstractmethod
    def sample_random_uniform(self, n):
        raise NotImplementedError

    @property
    def boundary(self):
        # Domain object of the boundary
        return ProductDomain(self.domain_a.boundary, self.domain_b.boundary)

    @property
    def inner(self):
        # open domain
        return ProductDomain(self.domain_a.inner, self.domain_b.inner)
