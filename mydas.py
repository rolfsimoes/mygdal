# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import mynumpy


def where(m, selection=None):
    result = mydas()
    if selection is None:
        for key in m.keys():
            result[key] = m.__data__[key].copy()
    else:
        for key in m.keys():
            result[key] = m.__data__[key][selection]
    return result


def orderby(m, key, reverse=False, result=None):
    if result is None:
        result = mydas()
    if not isinstance(key, str):
        raise TypeError('`key` is not string.')
    arg_sort = numpy.argsort(m[key], axis=-1)
    if reverse:
        arg_sort = arg_sort[::-1]
    for key in m.keys():
        result[key] = m.__data__[key][arg_sort]
    return result


def aggregate(m, values=None, funcs=None, groupby=None):
    """
    Aggregates `values` array's group values according to `funcs` function. The groups
    are formed by `groupby_keys` array's values. After aggregation, the resulting mydas
    has unique combined values in arrays `groupby_keys`.
    :param m: mydas
    :param values: Union[str, list, tuple]
    :param funcs: Union[function, callable object]
    :param groupby: Union[str, list, tuple]
    :return: mydas
    """
    result = mydas()
    if funcs is None:
        def funcs(x):
            return x
    if values is None:
        values = m.keys()
    if isinstance(values, str):
        values = (values,)
    if not isinstance(groupby, str):
        raise TypeError('`groupby` is not string.')
    if groupby:
        groups, groups_args = numpy.unique(m[groupby], return_inverse=True)
        result[groupby] = groups
        if isinstance(funcs, list) or isinstance(funcs, tuple):
            for i, key in enumerate(values):
                result[key] = numpy.array([funcs[i](m.__data__[key][numpy.where(groups_args == j)])
                                           for j in range(size(groups))])
        else:
            for key in values:
                result[key] = numpy.array([funcs(m.__data__[key][numpy.where(groups_args == i)])
                                           for i in range(size(groups))])
    else:
        if isinstance(funcs, list) or isinstance(funcs, tuple):
            for i, key in enumerate(values):
                result[key] = numpy.array([funcs[i](m.__data__[key])])
        else:
            for key in values:
                result[key] = numpy.array([funcs(m.__data__[key])])
    return result


def apply(m, keys, func, result=None):
    """
    Applies `func` on each individual value of `keys`'s arrays.
    Returns a copy of the `mydas` with modified arrays.
    :param m: mydas
    :param keys: Union[str, list, tuple]
    :param func: Union[function, callable object]
    :param result: mydas
    :return: mydas
    """
    if result is None:
        result = mydas()
    if isinstance(keys, str):
        keys = (keys,)
    for key in keys:
        _ = m.__data__[key]
    for key in m.keys():
        if key in keys:
            result[key] = numpy.array([func(value) for value in m.__data__[key]])
        else:
            result[key] = m.__data__[key].copy()
    return result


def append(m1, m2):
    if not (isinstance(m1, mydas) and isinstance(m2, mydas)):
        raise TypeError('one or both parameters are not mydas.')
    result = mydas()
    try:
        for key in m1.keys():
            result[key] = numpy.append(m1.__data__[key], m2.__data__[key], axis=0)
    except KeyError:
        raise ValueError('arguments have not same keys.')
    return result


def regularize(m, reg_key, slices, interp_keys=None, method='linear'):
    result = mydas()
    if interp_keys is None:
        interp_keys = m.keys()
    if isinstance(interp_keys, str):
        interp_keys = (interp_keys,)
    old_index = m.__data__[reg_key]
    new_index = mynumpy.linspace(min(old_index), max(old_index), slices, True)
    result[reg_key] = new_index
    for key in interp_keys:
        if key == reg_key:
            continue
        result[key] = mynumpy.interp(new_index, old_index, m.__data__[key], method=method)
    return result


def dtype(m, keys=None):
    if keys is None:
        keys = m.keys()
    if isinstance(keys, str):
        keys = (keys,)
    result = []
    for key in keys:
        result.append((key, m.__data__[key].dtype,
                       tuple() if isinstance(m.__data__[key], mydas) else m.__data__[key].shape[1:]))
    return numpy.dtype(result)


def shape(m, keys=None):
    if keys is None:
        keys = m.keys()
    if isinstance(keys, str):
        keys = (keys,)
    result = None
    for key in keys:
        if result is None:
            result = m.__data__[key].shape
        elif result != m.__data__[key].shape:
            raise TypeError('informed `keys` have not the same shape.')
    return result


def from_recarray(a, keys=None, selection=None):
    result = mydas()
    if keys is None:
        keys = a.dtype.names
    if isinstance(keys, str):
        keys = (keys,)
    if selection:
        a = a[selection]
    for key in keys:
        result[key] = a[key].copy()
    return result


def to_recarray(m, keys=None, selection=None):
    if keys is None:
        keys = m.keys()
    if isinstance(keys, str):
        keys = (keys,)
    result = None
    if selection is None:
        selection = numpy.ones(size(m), dtype='bool')
    for key in keys:
        if result is None:
            first_field = m.__data__[key][selection]
            result = numpy.empty(size(first_field), dtype=dtype(m, keys))
            if isinstance(first_field, mydas):
                result[key] = to_recarray(first_field, selection=selection)
            else:
                result[key] = first_field
        else:
            if isinstance(m.__data__[key], mydas):
                result[key] = to_recarray(m.__data__[key], selection=selection)
            else:
                result[key] = m.__data__[key][selection]
    return result


def compact(m, index_key, value_key):
    result = mydas()
    if isinstance(m, mydas):
        if isinstance(value_key, str):
            result[index_key] = m.__data__[index_key]
            result[value_key] = m.__data__[value_key]
        elif value_key:
            result[index_key] = m.__data__[index_key]
            result[value_key[0]] = numpy.empty((size(m), size(value_key)),
                                               dtype=m.__data__[value_key[0]].dtype)
            for i in range(size(value_key)):
                result[value_key[0]][:, i] = m.__data__[value_key[i]]
    elif m:
        if isinstance(value_key, str):
            result[index_key] = m[0].__data__[index_key]
            result[value_key] = numpy.empty((size(m[0]), size(m)),
                                            dtype=m[0].__data__[value_key].dtype)
            for i in range(size(m)):
                result[value_key][:, i] = m[i].__data__[value_key]
        else:
            raise TypeError('inconsistent arguments informed.')
    return result


def size(m):
    if isinstance(m, mydas):
        return m.__size__
    return len(m)


def head(obj, n=5):
    if type(obj) is mydas:
        print('{')
        if size(obj) > n:
            for key in obj.keys()[:n]:
                print('{}: '.format(key), end='')
                head(obj[key])
                print('...')
        else:
            for key in obj.keys():
                print('{}: '.format(key), end='')
                head(obj[key])
                print('')
        print('}')
    elif type(obj) is str:
        print(repr(obj), end=' ')
    elif type(obj) is numpy.datetime64:
        print("'{}'".format(obj), end=' ')
    elif type(obj) is tuple:
        print('(')
        if size(obj) > n:
            for i in range(n):
                head(obj[i])
                print('...')
        else:
            for i in range(size(obj)):
                head(obj[i])
                print('')
        print(')')
    elif type(obj) is list or type(obj) is numpy.ndarray:
        if size(obj) > n:
            print('[', end=' ')
            for i in range(n):
                head(obj[i])
            print('...]', end=' ')
        else:
            print('[', end=' ')
            for i in range(size(obj)):
                head(obj[i])
            print(']', end=' ')
    else:
        print(repr(obj), end=' ')


class mydas:
    def __init__(self, data=None):
        self.__size__ = 0
        self.__data__ = OrderedDict()
        if data is not None:
            for key in data.keys():
                self[key] = data[key]

    def __len__(self):
        return len(self.__data__)

    def __getitem__(self, index):
        """
        :param index: Union[str, Tuple[str, str, ...]]
        :return: Union[numpy.ndarray, mydas, Tuple[numpy.ndarray, ...]]
        """
        if isinstance(index, str):
            index = index
            return self.__data__[index]
        elif isinstance(index, slice) or isinstance(index, numpy.ndarray):
            result = mydas()
            for key in self.keys():
                result[key] = self.__data__[key][index]
            return result
        elif isinstance(index, tuple) or isinstance(index, list):
            return tuple(self.__data__[key] for key in index)
        else:
            raise TypeError('index is not `str`, `slice`, `numpy.ndarray`, `tuple`, or `list`.')

    def __setitem__(self, key, value):
        """
        :param key: str
        :param value: iterable
        :return: None
        """
        if not key:
            return
        if not isinstance(key, str):
            raise TypeError('`key` is not string.')
        if len(self) and size(value) != size(self):
            raise ValueError('new data has not same __size__.')
        if isinstance(value, mydas):
            self.__data__[key] = value.copy()
        else:
            self.__data__[key] = numpy.asarray(value).copy()
        if not size(self):
            self.__size__ = size(value)

    def __delitem__(self, key):
        self.__data__.__delitem__(key)

    def __repr__(self):
        content = ''
        for key in self.keys():
            content += '{}: {}\n'.format(key, repr(self.__data__[key]))
        return '{\n' + content + '}'

    def __iter__(self):
        return iter(self.values())

    @property
    def dtype(self):
        return dtype(m=self, keys=self.keys())

    @property
    def shape(self):
        return shape(self, self.keys())

    def values(self):
        return tuple(self[key] for key in self.keys())

    def keys(self):
        return tuple(self.__data__.keys())

    def size(self):
        return self.__size__

    def copy(self):
        result = mydas()
        for key in self.keys():
            result[key] = self.__data__[key].copy()
        return result

    def where(self, selection=None):
        return where(m=self, selection=selection)

    def orderby(self, keys, reverse=False):
        orderby(m=self, key=keys, reverse=reverse, result=self)

    def aggregate(self, values=None, funcs=None, groupby=None):
        return aggregate(m=self, values=values, funcs=funcs, groupby=groupby)

    def apply(self, keys, func):
        apply(m=self, keys=keys, func=func, result=self)

    def append(self, other):
        return append(m1=self, m2=other)

    def regularize(self, reg_key, slices, interp_keys=None, method='linear'):
        return regularize(m=self, reg_key=reg_key, slices=slices, interp_keys=interp_keys, method=method)
