"""The basic structure of every sampler and all sampler 'operations'.
"""
import abc
import numpy as np


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

    def append(self, other):
        assert isinstance(other, DataSampler)
        assert len(other) == len(self)
        return AppendSampler(self, other)


class ProductSampler(DataSampler):
    def __init__(self, sampler_a, sampler_b):
        self.sampler_a = sampler_a
        self.sampler_b = sampler_b
        super().__init__(len(self.sampler_a) * len(self.sampler_b))

    def sample_points(self):
        b_points = self.sampler_b.sample_points()
        
        for b in b_points:
            self.sampler_a.update(b)

        return super().sample_points()


class ConcatSampler(DataSampler):
    """A sampler that adds two single samplers together.
    Will concatenate the data points of both samplers.

    Parameters
    ----------
    sampler_a, sampler_b : DataSampler
        The two DataSamplers that should be connected.
    """
    def __init__(self, sampler_a, sampler_b):
        self.sampler_a = sampler_a
        self.sampler_b = sampler_b
        super().__init__(len(self.sampler_a) + len(self.sampler_b))

    def sample_points(self):
        samples_a = self.sampler_a.sample_points()
        samples_b = self.sampler_b.sample_points()
        for vname in samples_a:
            samples_a[vname] = np.concatenate((samples_a[vname], 
                                               samples_b[vname]), 
                                              axis=0)
        return samples_a


class AppendSampler(DataSampler):
    """A sampler that appends the output of two single samplers together.

    Parameters
    ----------
    sampler_a, sampler_b : DataSampler
        The two DataSamplers that should be connected. Need two have the same 
        length.
    """
    def __init__(self, sampler_a, sampler_b):
        self.sampler_a = sampler_a
        self.sampler_b = sampler_b
        super().__init__(len(self.sampler_a))

    def sample_points(self):
        # if callable(sampler_a):
        # ....
        #
        samples_a = self.sampler_a.sample_points()
        samples_b = self.sampler_b.sample_points()
        return  {**samples_a, **samples_b}