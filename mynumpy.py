import numpy


# datetime routines
def datetime64_to_timestamp(datetime64):
    return (datetime64 - numpy.datetime64('1970-01-01', 'D')) / numpy.timedelta64(1, 's')


def timestamp_to_datetime64(timestamp):
    return numpy.array(timestamp * numpy.timedelta64(1, 's') + numpy.datetime64('1970-01-01', 'D'),
                       dtype='datetime64[D]')


def datetime64_unit(datetime64):
    result = str(datetime64.dtype)
    return result[datetime64.index('[') + 1:datetime64.index(']')]


def datetime64_doy(datetime64, base_date=None):
    if base_date is None:
        base_date = numpy.array(datetime64, dtype='datetime64[D]')
    return (datetime64 - base_date) / numpy.timedelta64(1, 'D')


# arrays routines
def arange(start, stop, step, step_unit='D', array_unit=None, limit=65536):
    if isinstance(start, numpy.datetime64) or isinstance(stop, numpy.datetime64):
        if (stop - start) / (step * numpy.timedelta64(1, step_unit)) > limit:
            raise ValueError('Number of index to be generated exceeds the limit.')
        result = numpy.arange(start, stop, step * numpy.timedelta64(1, step_unit))
    elif step_unit is not None and array_unit is not None:
        if (stop - start) / (step * numpy.timedelta64(1, step_unit) / numpy.timedelta64(1, array_unit)) > limit:
            raise ValueError('Number of index to be generated exceeds the limit.')
        result = numpy.arange(start, stop, step * numpy.timedelta64(1, step_unit) / numpy.timedelta64(1, array_unit))
    else:
        raise TypeError("Inconsistent parameters' values informed.")
    return result


def linspace(start, stop, num, endpoint=True):
    if isinstance(start, numpy.datetime64):
        result = timestamp_to_datetime64(numpy.linspace(datetime64_to_timestamp(start),
                                                        datetime64_to_timestamp(stop), num, endpoint=endpoint))
    else:
        result = numpy.linspace(start, stop, num, endpoint=endpoint)
    return result


def interp(x, xp, fp, left=None, right=None, period=None, method='linear'):
    if method == 'linear':
        idx_from = xp[0]
        if type(idx_from) is numpy.datetime64:
            result = numpy.interp(datetime64_to_timestamp(x), datetime64_to_timestamp(xp), fp, left, right, period)
        else:
            result = numpy.interp(x, xp, fp, left, right, period)
    elif method == 'nearest' and period is None:
        new_argindex_last = numpy.searchsorted(xp, x, 'right') - 1
        new_argindex_next = numpy.searchsorted(xp, x, 'left')
        xp = numpy.append(xp, (left if left is not None else x[0], right if right is not None else x[-1]))
        fp = numpy.append(fp, (left if left is not None else fp[0], right if right is not None else fp[-1]))
        new_argindex_last[new_argindex_last == -1] = -2
        new_argindex_next[new_argindex_next == len(xp)] = -1
        dist_last = numpy.abs(xp[new_argindex_last] - x)
        dist_next = numpy.abs(xp[new_argindex_next] - x)
        result = fp[numpy.where(dist_last <= dist_next, new_argindex_last, new_argindex_next)]
    elif method == 'last' and right is None and period is None:
        new_argindex_last = numpy.searchsorted(xp, x, 'right') - 1
        fp = numpy.append(fp, left if left is not None else fp[0])
        result = fp[new_argindex_last]
    elif method == 'next' and left is None and period is None:
        new_argindex_next = numpy.searchsorted(xp, x, 'left')
        fp = numpy.append(fp, right if right is not None else fp[-1])
        result = fp[new_argindex_next]
    else:
        raise TypeError("Inconsistent parameters' values informed.")
    return result
