from .domain import Domain
from ...utils.user_fun import UserFunction
import inspect
import functools


class LambdaDomain(Domain):
    def __init__(self, constructor, params, space, dim, necessary_variables=None):
        """
        params : A dictionary containing all the params needed to create the given
            domain_class domain
        """
        super().__init__(space, dim=dim)
        self.constructor = constructor
        self.params = params

        # create a set of variables/spaces that this domain needs to be properly defined
        if necessary_variables is not None:
            self.necessary_variables = necessary_variables
        else:
            self.necessary_variables = set()
            for key in params:
                if callable(params[key]):
                    if '*' in str(inspect.signature(params[key])):
                        raise ValueError("""Functions in Domain definitions should use proper keys,
                                            as defined in their spaces.""")
                    params[key] = UserFunction(params[key])
                    for k in params[key].necessary_args:
                        self.necessary_variables.add(k)

    def __call__(self, **data):
        """
        Slice the domain along given axis, i.e. for given (partial) data
        """

    def _call_param(param, args):
        if callable(param):
            if all(arg in args for arg in param.necessary_args):
                return param(**args)
            else:
                param.set_default(**args)
        return param

    def __mul__(self, other):
        if not isinstance(other, LambdaDomain) and all(var in other.space for var in self.necessary_variables):
            return LambdaProductDomain(self, other)
        if isinstance(other, LambdaDomain):
            return LambdaDomain

    def _get_necessary_args(fun):
        """
        Returns the (positional or keyword-)arguments of fun which don't supply
        default values
        NOTE: we need a whole library utils part for user-function handling
        """



class LambdaProductDomain(ProductDomain):
    def __init__(self, other):
        super().__init__()
        if isinstance(self, LambdaDomain):

    def is_inside(self, points):
        return super().is_inside(points)

    def bounding_box(self):
        # NOTE: This is only an estimate on the bounding box
        return super().bounding_box()

    def __add__(self, other):
        pass

    def __or__(self, other):
        # Union
        return self.__add__(other)

    def __sub__(self, other):
        # Difference
        pass

    def __and__(self, other):
        # Intersection
        pass