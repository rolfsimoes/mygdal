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


def interp(x, xp, fp, left=None, right=None, period=None):
    idx_from = xp[0]
    if type(idx_from) is numpy.datetime64:
        result = numpy.interp(datetime64_to_timestamp(x), datetime64_to_timestamp(xp), fp, left, right, period)
    else:
        result = numpy.interp(x, xp, fp, left, right, period)
    return result
