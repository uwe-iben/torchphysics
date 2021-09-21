from typing import NewType
import warnings
import abc
from collections import Counter


class Space(Counter):

    def __init__(self, variables_dims):
        # set counter of variable names and their dimensionalities
        super.__init__(variables_dims)

    def __mul__(self, other):
        assert isinstance(other, Space)
        return Space(self + other)

    @property
    def dim(self):
        return sum(self.values())


class R1(Space):
    def __init__(self, variable_name):
        super().__init__({variable_name: 1})


class R2(Space):
    def __init__(self, variable_name):
        super().__init__({variable_name: 2})


class R3(Space):
    def __init__(self, variable_name):
        super().__init__({variable_name: 3})


X = R1('x')
T = R1('t')
S = X * T


class Domain:
    def __init__(self, space, dim=None):
        self.space = space
        if dim is None:
            self.dim = self.space.dim
        else:
            self.dim = dim

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


class BoundaryDomain(Domain):
    def __init__(self, domain, dim):
        super().__init__(domain.space, dim=domain.dim-1)


class Domain3D(Domain):
    def __init__(self, mesh, space):
        self.mesh = mesh
        super().__init__(space, dim=3)


class Domain2D(Domain):
    def __init__(self, polygon, space):
        self.polygon = polygon
        super().__init__(space, dim=3)


class ProductDomain(Domain):
    def __init__(self, domain_a, domain_b):
        if not domain_a.space.keys().isdisjoint(domain_b.space):
            warnings.warn("""Warning: The space of a ProductDomain will be the product
                of its factor domains spaces. This may lead to unexpected behaviour.""")
        space = domain_a.space + domain_b.space
        super().__init__(space)

    def is_inside(self, points):
        return super().is_inside(points)

    def bounding_box(self):
        return super().bounding_box()


# use shapely or trimesh as much as possible

class Point(Domain):
    pass


class Interval(Domain):
    pass


class Polygon(Domain):
    pass


class Polyeder(Domain):
    pass


I = Interval(T, [0, 1])
R = Interval(X, lambda t: [0, 2**(-t)])
D = R * I
# evaluate all Lambda functions in ProductDomain


class DataSampler:

    def __init__(self, n_points):
        self.n_points = n_points

    def __len__(self):
        return self.n_points

    @abc.abstractmethod
    def sample_points(self):
        raise NotImplementedError

    def __mul__(self, other):
        assert isinstance(other, DataSampler)
        # returns a sampler that samples from the 'cartesian product'
        # of the samples of two samplers
        return ProductSampler(self, other)

    def __add__(self, other):
        assert isinstance(other, DataSampler)
        # returns a sampler that samples from two samplers
        return ConcatSampler(self, other)

class ProductSampler(DataSampler):
    def __init__(self, sampler_a, sampler_b):
        self.sampler_a = sampler_a
        self.sampler_b = sampler_b
        super().__init__(len(self.sampler_a) * len(self.sampler_b))
    
    def sample_points(self):
        return super().sample_points()


class ConcatSampler(DataSampler):
    def __init__(self, sampler_a, sampler_b):
        self.sampler_a = sampler_a
        self.sampler_b = sampler_b
        super().__init__(len(self.sampler_a) + len(self.sampler_b))

    def sample_points(self):
        samples_a = self.sampler_a.sample_points()
        samples_b = self.sampler_b.sample_points()


class Condition(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
