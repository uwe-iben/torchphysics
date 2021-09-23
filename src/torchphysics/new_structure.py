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
    def __init__(self, space, dim=None, tol=1e-6):
        self.space = space
        if dim is None:
            self.dim = self.space.dim
        else:
            self.dim = dim
        self.tol = tol

    @abc.abstractmethod
    @property
    def is_initialized(self):
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
        # Cartesian product
        return ProductDomain(self, other)

    def __add__(self, other):
        # Union
        raise NotImplementedError

    def __or__(self, other):
        # Union
        return self.__add__(other)

    def __sub__(self, other):
        # Difference
        raise NotImplementedError

    def __and__(self, other):
        # Intersection
        raise NotImplementedError

    @property
    def boundary(self):
        # Domain object of the boundary
        raise NotImplementedError

    @property
    def inner(self):
        # open domain
        raise NotImplementedError


class LambdaDomain(Domain):
    def __init__(self, class_, params, space, dim):
        super().__init__(space, dim=dim)
        self.class_ = class_
        self.params = params

    def __call__(self, data):
        p = {}
        for k in self.params:
            if callable(self.params[k]):
                p[k] = self.params[k](data)
            else:
                p[k] = self.params[k]
        return self.class_(space=self.space, **p)

    def __mul__(self, other):
        return super().__mul__(other)


class BoundaryDomain(Domain):
    def __init__(self, domain, dim):
        super().__init__(domain.space, dim=domain.dim-1)

    @abc.abstractmethod
    def normal(self, points):
        pass


class Domain3D(Domain):
    def __init__(self, mesh, space):
        self.mesh = mesh
        assert space.dim == 3
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
        space = domain_a.space * domain_b.space
        super().__init__(space, dim=domain_a.dim + domain_b.dim)

    def is_inside(self, points):
        return 

    def bounding_box(self):
        return 

    def sample_grid(self, n):
        return super().sample_grid(n)

    def sample_random_uniform(self, n):
        return super().sample_random_uniform(n)


class LambdaProductDomain(ProductDomain):
    def is_inside(self, points):
        return super().is_inside(points)

    def bounding_box(self):
        return super().bounding_box()
    
    def __add__(self, other):



# use shapely or trimesh as much as possible

class Point(Domain):
    def __init__(self, coord, space):
        super().__init__(space, dim=0)
        if callable(coord):
            return LambdaDomain(Point, {'coord': coord}, space=space, dim=0)


class Interval(Domain):
    pass


class Polygon(Domain):
    pass


class Polyeder(Domain):
    pass


I = Interval(T, [0, 1])
R = Interval(X, lambda t: [0, 2**(-t)])
G = Circle(Y, center = lambda t: [0,t], radius = 1)
D = R * I
E = G * D
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
        a_points = sampler_a.sample_points()
        for a in a_points:
            sampler_b.update(a)

        return super().sample_points()


class ConcatSampler(DataSampler):
    def __init__(self, sampler_a, sampler_b):
        self.sampler_a = sampler_a
        self.sampler_b = sampler_b
        super().__init__(len(self.sampler_a) + len(self.sampler_b))

    def sample_points(self):
        samples_a = self.sampler_a.sample_points()
        samples_b = self.sampler_b.sample_points()


class GridSampler(DataSampler):
    def __init__(self, domain, n_points):
        super().__init__(n_points)
        self.domain = domain

    def sample_points(self):
        return self.domain.sample_grid(len(self))


class RandomUniformSampler(DataSampler):
    pass


class GaussianSampler:
    pass

sampler_a = GridSampler(domain=T.boundary, n_points=1)
sampler_b = RandomUniformSampler(domain=X, n_points=100)
t_dirichlet_sampler = sampler_a * sampler_b


class Condition(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
