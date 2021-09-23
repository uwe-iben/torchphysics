import abc
import numpy as np


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

    @abc.abstractmethod
    def __call__(self, **kwargs):
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