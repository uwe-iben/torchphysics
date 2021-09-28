import numpy as np
import warnings

from .domain import Domain, BoundaryDomain
from .lambdadomain import LambdaDomain
from .domain0D import Point, PointCloud


class Interval(Domain):
    """Creates a Interval of the form [a, b].

    Parameters
    ----------
    space : Space
        The space in which this object lays.
    lower_bound : Number or callable
        The left/lower bound of the interval.
    upper_bound : Number or callable
        The right/upper bound of the interval.
    """
    def __new__(cls, space, lower_bound, upper_bound, tol=1e-06):
        if callable(lower_bound) or callable(upper_bound):
            params = {'lower_bound': lower_bound, 'upper_bound': upper_bound}
            return LambdaDomain(constructor=cls, params=params, space=space, dim=1, tol=tol)
        return super(Interval, cls).__new__(cls)

    def __init__(self, space, lower_bound, upper_bound, tol=1e-06):
        assert lower_bound < upper_bound
        super().__init__(space, dim=1, tol=tol)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def is_inside(self, points):
        points = super()._check_single_point(points)
        return ((self.lower_bound-self.tol < points[:])
                & (points[:] < self.upper_bound+self.tol)).reshape(-1, 1)

    def bounding_box(self):
        return [self.lower_bound, self.upper_bound]

    def sample_random_uniform(self, n):
        points = np.random.uniform(self.lower_bound, self.upper_bound, (n, 1))
        return points.astype(np.float32)

    def sample_grid(self, n):
        points = np.linspace(self.lower_bound, self.upper_bound, n+2)[1:-1]
        return points.astype(np.float32).reshape(-1, 1)

    @property
    def boundary(self):
        """Returns both boundary points of the domain.
        """
        return BoundaryDomain1D(self, points=[[self.lower_bound], [self.upper_bound]])

    @property
    def boundary_left(self):
        """Returns only the left boundary of the domain.
        """
        return BoundaryDomain1D(self, points=self.lower_bound)

    @property
    def boundary_right(self):
        """Returns only the right boundary of the domain.
        """
        return BoundaryDomain1D(self, points=self.upper_bound)

    def __add__(self, other):
        assert other.dim == 1
        if isinstance(other, IntervalColletion):
            return other + self
        # If the two intervals touch each other we get a single new interval
        new_tol = np.min([self.tol, other.tol])
        if any(self._other_interval_inside(other)):
            min_bound = np.min([self.lower_bound, other.lower_bound])
            max_bound = np.max([self.upper_bound, other.upper_bound])
            return Interval(space=self.space, lower_bound=min_bound,
                            upper_bound=max_bound, tol=new_tol)
        else:  # return a new object consisting of two intervals
            return IntervalColletion(self.space, intervals=[self, other], tol=new_tol)

    def __sub__(self, other):
        assert other.dim == 1
        if isinstance(other, IntervalColletion):
            return self._cut_interval_with_collection(other)
        return self._cut_two_intervals(self, other)

    def _cut_interval_with_collection(self, collection_):
        # cut the base interval with every interval of the collection
        interval_1 = self
        for i in collection_.intervals:
            interval_1 = interval_1 - i
            if interval_1 is None:
                warnings.warn("""After the cut the interval is empty!""")
                break
        return interval_1

    def _cut_two_intervals(self, interval_1, interval_2):
        # first check if whole interval gets deleted
        if all(interval_2._other_interval_inside(interval_1)):
            warnings.warn("""After the cut the interval is empty!""")
            return
        # check different cases
        new_tol = np.min([interval_1.tol, interval_2.tol])
        bounds_inside = interval_1._other_interval_inside(interval_2)
        if any(bounds_inside):
            new_1, new_2 = self._create_new_cut_intervals(interval_1, interval_2,
                                                          new_tol, bounds_inside)
            if all(bounds_inside):  # cut in the middle
                return IntervalColletion(self.space, [new_1, new_2], tol=new_tol)
            elif bounds_inside[0]:  # cut a piece away
                return new_1
            else:  # cut a piece away
                return new_2
        return interval_1

    def _create_new_cut_intervals(self, interval_1, interval_2,
                                  new_tol, bounds_inside):
        new_1, new_2 = None, None
        if bounds_inside[0]:
            new_1 = Interval(self.space, interval_1.lower_bound,
                             interval_2.lower_bound, tol=new_tol)
        if bounds_inside[1]:
            new_2 = Interval(self.space, interval_2.upper_bound,
                             interval_1.upper_bound, tol=new_tol)
        return new_1, new_2

    def __and__(self, other):
        assert other.dim == 1
        if isinstance(other, IntervalColletion):
            return other & self
        min_bound = np.max([self.lower_bound, other.lower_bound])
        max_bound = np.min([self.upper_bound, other.upper_bound])
        return Interval(space=self.space, lower_bound=min_bound,
                        upper_bound=max_bound, tol=np.min([self.tol, other.tol]))

    def _other_interval_inside(self, other):
        return self.is_inside(other.bounding_box())


class IntervalColletion(Domain):
    """Handels the case of disjoint intervals, that can get created while
    cutting/unitting different intervals.

    Parameters
    ----------
    space : Space
        The space in which this object lays.
    intervals : list of Interval
        All disjoint intervals.
    """

    def __init__(self, space, intervals, tol=1e-06):
        super().__init__(space, dim=1, tol=tol)
        self.intervals = intervals
        self._compute_bounds_and_length()

    def _compute_bounds_and_length(self):
        bounds = np.zeros((len(self.intervals), 2))
        self.length = 0
        index = 0
        for i in self.intervals:
            bounds[index] = i.bounding_box()
            self.length += (bounds[index][1] - bounds[index][0])
            index += 1
        self.lower_bound = np.min(bounds[:, 0])
        self.upper_bound = np.max(bounds[:, 1])

    def is_inside(self, points):
        inside = np.zeros((len(points), 1), dtype=bool)
        for i in self.intervals:
            in_i = i.is_inside(points)
            index = np.where(in_i)[0]
            inside[index] = True
        return inside

    def bounding_box(self):
        return [self.lower_bound, self.upper_bound]

    def sample_random_uniform(self, n):
        points = np.zeros((n, 1))
        current_n, counter = 0, 0
        for i in self.intervals:
            scaled_n = self._scale_number_of_points(n, current_n, counter, i)
            new_points = i.sample_random_uniform(scaled_n)
            points[range(current_n, current_n+scaled_n)] = new_points
            current_n += scaled_n
            counter += 1
        return points.astype(np.float32)

    def sample_grid(self, n):
        points = np.zeros((n, 1))
        current_n, counter = 0, 0
        for i in self.intervals:
            scaled_n = self._scale_number_of_points(n, current_n, counter, i)
            points[range(current_n, current_n+scaled_n)] = i.sample_grid(scaled_n)
            current_n += scaled_n
            counter += 1
        return points.astype(np.float32)

    def _scale_number_of_points(self, n, current_n, counter, i):
        length = i.upper_bound - i.lower_bound
        if counter == len(self.intervals) - 1:
            scaled_n = n - current_n
        else:
            scaled_n = int(n * length/self.length)
        return scaled_n

    @property
    def boundary(self):
        bounds = np.zeros((len(self.intervals), 2))
        for i in range(len(self.intervals)):
            bounds[i] = self.intervals[i].bounding_box()
        return BoundaryDomain1D(self, points=bounds.flatten().reshape(-1, 1))

    @property
    def boundary_left(self):
        return BoundaryDomain1D(self, points=self.lower_bound)

    @property
    def boundary_right(self):
        return BoundaryDomain1D(self, points=self.upper_bound)

    def __add__(self, other):
        assert other.dim == 1
        other = self._change_other_to_collection(other)
        self.intervals.extend(other.intervals)
        return self._connect_intervals()

    def _connect_intervals(self):
        new_intervals = []
        for i in self.intervals:
            i = self.check_overlap(i, self.intervals)
            i = self.check_overlap(i, new_intervals)
            new_intervals.append(i)
        return self._new_collection(new_intervals)

    def check_overlap(self, i, interval_list):
        for k in interval_list[:]:
            if not i == k:
                if any(i._other_interval_inside(k)):
                    i += k
                    interval_list.remove(k)
        return i

    def __sub__(self, other):
        assert other.dim == 1
        other = self._change_other_to_collection(other)
        new_intervals = []
        for i in self.intervals:
            new_inter = i - other
            if new_inter is not None:
                new_intervals.append(new_inter)
        return self._new_collection(new_intervals)

    def _new_collection(self, new_intervals):
        if len(new_intervals) == 0:
            warnings.warn("""After the operation the intervals were empty!""")
            return
        elif len(new_intervals) == 1:
            return new_intervals[0]
        else:
            return IntervalColletion(self.space, new_intervals, self.tol)

    def _change_other_to_collection(self, other):
        if isinstance(other, Interval):
            other = IntervalColletion(space=other.space,
                                      intervals=[other], tol=other.tol)
        return other

    def __and__(self, other):
        assert other.dim == 1
        other = self._change_other_to_collection(other)
        for i in range(len(self.intervals)):
            for other_inter in other.intervals:
                bounds_inside = self.intervals[i]._other_interval_inside(other_inter)
                if any(bounds_inside):
                    self.intervals[i] &= other_inter
        return self


class BoundaryDomain1D(BoundaryDomain):
    """Handels the boundary of intervals.
    """

    def __init__(self, domain, points):
        super().__init__(domain)
        self.domain = domain
        if isinstance(points, (list, np.ndarray)):
            self.point_object = PointCloud(space=domain.space,
                                           coord_list=points, tol=domain.tol)
        else:
            self.point_object = Point(space=domain.space,
                                      coord=points, tol=domain.tol)

    def is_inside(self, points):
        return self.point_object.is_inside(points)

    def bounding_box(self):
        return self.domain.bounding_box()

    def sample_grid(self, n):
        return self.point_object.sample_grid(n)

    def sample_random_uniform(self, n):
        return self.point_object.sample_random_uniform(n)

    def normal(self, points):
        points = super()._check_single_point(points)
        normals = np.ones_like(points)
        index_left = self._get_index_left(points)
        normals[index_left] *= -1
        return normals.astype(np.float32)

    def _get_index_left(self, points):
        if isinstance(self.domain, Interval):
            dist = np.linalg.norm(points - self.domain.lower_bound, axis=1)
            return np.where(dist <= self.tol)[0]
        else:  # IntervalCollection
            index = []
            for i in self.domain.intervals:
                dist = np.linalg.norm(points - i.lower_bound, axis=1)
                index.extend(np.where(dist <= self.tol)[0])
            return index
