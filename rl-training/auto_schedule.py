from ray.rllib.execution.common import _get_global_vars
from copy import copy, deepcopy
import numpy as np

class AutoSchedule:

    def __init__(self, schedule):
        self._schedule = schedule

    @property
    def val(self):
        # try:
        steps_sampled = _get_global_vars()["timestep"]
        # except:
        #     return np.nan

        return self._schedule(steps_sampled)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def __float__(self):
        return self.val


def make_func(name):
    return lambda self, *args: getattr(self.val, name)(*args)

exclude = [ '__class__', '__delattr__', '__dir__', '__divmod__', '__doc__', '__eq__', '__float__', '__floordiv__', '__format__', '__getattribute__', '__getformat__', '__getnewargs__', '__hash__', '__init__', '__init_subclass__', '__new__', '__pos__','__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rtruediv__', '__set_format__', '__setattr__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__trunc__', 'as_integer_ratio', 'conjugate', 'fromhex', 'hex', 'imag', 'is_integer', 'real']

for name in [a for a in dir(float) if a not in exclude]:
    setattr(AutoSchedule, name, make_func(name))
