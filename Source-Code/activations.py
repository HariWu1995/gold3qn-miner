import math

from keras import backend as K
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects


def gelu(x):
    """
    An approximation of gelu.
    See: https://arxiv.org/pdf/1606.08415.pdf
    """
    x_pow_3 = x * x * x
    act = 0.5 * x * (1.0+K.tanh(math.sqrt(2.0/math.pi) * (x+0.044715*x_pow_3)))
    return act


class GELU(Activation):
    def __init__(self, activation, **kwargs):
        super(GELU, self).__init__(activation, **kwargs)
        self.__name__ = 'GELU'


def swish(x, beta=1):
    return x * K.sigmoid(beta*x)


class SWISH(Activation):
    def __init__(self, activation, **kwargs):
        super(SWISH, self).__init__(activation, **kwargs)
        self.__name__ = 'SWISH'


get_custom_objects().update({'swish': SWISH(swish)})
get_custom_objects().update({'gelu': GELU(gelu)})

