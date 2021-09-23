from .domain import Domain


class LambdaDomain(Domain):
    def __init__(self, class_, params, space, dim):
        super().__init__(space, dim=dim)
        self.class_ = class_
        self.params = params

    def __call__(self, **data):
        for variable in data:
            
        p = {}
        for k in self.params:
            if callable(self.params[k]):
                p[k] = self.params[k](data)
            else:
                p[k] = self.params[k]
        return self.class_(space=self.space, **p)

    def __mul__(self, other):
        return LambdaProductDomain(self, other)


class LambdaProductDomain(ProductDomain):
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