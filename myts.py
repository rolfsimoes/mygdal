import numpy
import mynumpy
import scipy
import scipy.sparse.linalg
from matplotlib import cm


def doy_index(index, base_date=None):
    return (index - (base_date if base_date else numpy.array(index, dtype='datetime64[D]'))) / numpy.timedelta64(1, 'D')


def pick_colors(cmap_str, categories, include_extrems=False):
    result = {}
    cmap = cm.get_cmap(cmap_str)
    unique_categories = numpy.unique(categories)
    n = len(unique_categories)
    increment = 0 if include_extrems else 1
    for i, value in enumerate(unique_categories):
        result[value] = cmap((i + increment) / (n + increment))
    return result


class MyTS:
    def __init__(self, index):
        if not (isinstance(index, float) or isinstance(index, numpy.datetime64)):
            raise ValueError('Index is not timestamp or datetime64.')
        self.__idx__ = index
        self.__series__ = []

    def __len__(self):
        return len(self.__idx__)

    @property
    def index(self):
        return self.__idx__

    @property
    def series(self):
        return self.__series__

    def append_measures(self, x):
        if len(x) != len(self):
            raise ValueError("Values has not same length as 'self'.")
        self.__series__.append(numpy.asarray(x))

    def remove_measures(self, i):
        return self.__series__.pop(i)


def new(index, measures=None):
    result = MyTS(index)
    if measures:
        result.append_measures(measures)
    return result


def compact(myts, func):
    result = MyTS(numpy.unique(myts.index))
    for measures in myts.series:
        result.append_measures(numpy.array([func(measures[numpy.where(myts.index == value)])
                                            for value in result.series]))
    return result


def ts_between_args(ts, idx_from, idx_to, include_extrems=False):
    result = (idx_from <= ts_index(ts)) * (ts_index(ts) <= idx_to)
    if include_extrems:
        result[1:] = result[1:] + result[:-1]
        result[:-1] = result[:-1] + result[1:]
    return result


def ts_between(ts, idx_from, idx_to, include_extrems=False):
    interval = ts_between_args(ts, idx_from, idx_to, include_extrems)
    return ts_new(ts_index(ts)[interval], ts_values(ts)[interval])


def ts_regular(ts, n=None, endpoint=None, step=None, unit=None, index=None):
    idx_from, idx_to = ts_index(ts)[0], ts_index(ts)[-1]
    if n is not None and step is None and unit is None and index is None:
        endpoint = True if endpoint is None else endpoint
        new_index = linspace_index(idx_from, idx_to, n, endpoint=endpoint)
        new_values = interp_values(new_index, ts_index(ts), ts_values(ts))
    elif step is not None and n is None and endpoint is None and index is None:
        unit = 'D' if unit is None else unit
        new_index = range_index(idx_from, idx_to, step, unit)
        new_values = interp_values(new_index, ts_index(ts), ts_values(ts))
    elif index is not None and n is None and endpoint is None and step is None and unit is None:
        new_index = index[(index >= idx_from) * (index <= idx_to)]
        new_values = interp_values(new_index, ts_index(ts), ts_values(ts))
    else:
        raise TypeError('Incompatible parameters informed.')
    result = ts_new(new_index, new_values)
    return result


def ts_index_doy(ts, base_date=None):
    return ts_new(doy_index(ts_index(ts), base_date), ts_values(ts))


def ts_smooth(ts, lmda=1.0):
    # This is a function that implements a Whittaker smoother in Python.
    # Reference: Paul H. C. Eilers. "A Perfect Smoother". Analytical Chemistry, 2003, 75 (14), pp 3631–3636.
    # Source: https://gist.github.com/zmeri/3c43d3b98a00c02f81c2ab1aaacc3a49

    def whittaker_smooth(y, lmda):
        m = len(y)
        E = scipy.sparse.identity(m)
        d1 = -1 * numpy.ones((m), dtype='d')
        d2 = 3 * numpy.ones((m), dtype='d')
        d3 = -3 * numpy.ones((m), dtype='d')
        d4 = numpy.ones((m), dtype='d')
        D = scipy.sparse.diags([d1, d2, d3, d4], [0, 1, 2, 3], shape=(m - 3, m), format="csr")
        z = scipy.sparse.linalg.cg(E + lmda * (D.transpose()).dot(D), y)
        return z[0]

    return ts_new(ts_index(ts), whittaker_smooth(ts_values(ts), lmda))