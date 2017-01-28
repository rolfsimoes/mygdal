# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import mynumpy


def where(myda, selection=None):
    result = MyDA()
    if selection is None:
        for key in myda.keys():
            result[key] = myda.__data__[key]
    else:
        for key in myda.keys():
            result[key] = myda.__data__[key][selection]
    return result


def orderby(myda, key, reverse=False, result=None):
    if result is None:
        result = MyDA()
    if not isinstance(key, str):
        raise TypeError('`key` is not string.')
    arg_sort = numpy.argsort(myda[key], axis=-1)
    if reverse:
        arg_sort = arg_sort[::-1]
    for key in myda:
        result[key] = myda.__data__[key][arg_sort]
    return result


def aggregate(myda, values=None, funcs=None, groupby=None):
    """
    Aggregates `values` array's group values according to `funcs` function. The groups
    are formed by `groupby_keys` array's values. After aggregation, the resulting MyDA
    has unique combined values in arrays `groupby_keys`.
    :param myda: MyDA
    :param values: Union[str, list, tuple]
    :param funcs: Union[function, callable object]
    :param groupby: Union[str, list, tuple]
    :return: MyDA
    """
    result = MyDA()
    if funcs is None:
        def funcs(x):
            return x
    if values is None:
        values = myda.keys()
    if isinstance(values, str):
        values = (values,)
    if not isinstance(groupby, str):
        raise TypeError('`groupby` is not string.')
    if groupby:
        groups, groups_args = numpy.unique(myda[groupby], return_inverse=True)
        if not isinstance(groupby, str):
            groupby = '_'.join(groupby)
        result[groupby] = groups
        if isinstance(funcs, list) or isinstance(funcs, tuple):
            for i, key in enumerate(values):
                result[key] = numpy.array([funcs[i](myda.__data__[key][numpy.where(groups_args == j)])
                                           for j in range(len(groups))])
        else:
            for key in values:
                result[key] = numpy.array([funcs(myda.__data__[key][numpy.where(groups_args == i)])
                                           for i in range(len(groups))])
    else:
        if isinstance(funcs, list) or isinstance(funcs, tuple):
            for i, key in enumerate(values):
                result[key] = numpy.array([funcs[i](myda.__data__[key])])
        else:
            for key in values:
                result[key] = numpy.array([funcs(myda.__data__[key])])
    return result


def apply(myda, keys, func, result=None):
    """
    Applies `func` on each individual value of `keys`'s arrays.
    Returns a copy of the `myda` with modified arrays.
    :param myda: MyDA
    :param keys: Union[str, list, tuple]
    :param func: Union[function, callable object]
    :param result: MyDA
    :return: MyDA
    """
    if result is None:
        result = MyDA()
    if isinstance(keys, str):
        keys = (keys,)
    for key in keys:
        _ = myda.__data__[key]
    for key in myda.keys():
        if key in keys:
            result[key] = numpy.array([func(value) for value in myda.__data__[key]])
        else:
            result[key] = myda.__data__[key]
    return result


def append(myda1, myda2):
    if not (isinstance(myda1, MyDA) and isinstance(myda2, MyDA)):
        raise TypeError('one or both parameters are not MyDA.')
    result = MyDA()
    try:
        for key in myda1:
            result[key] = numpy.append(myda1.__data__[key], myda2.__data__[key], axis=0)
    except KeyError:
        raise ValueError('arguments have not same keys.')
    return result


def regularize(myda, reg_key, slices, interp_keys=None, method='linear'):
    result = MyDA()
    if interp_keys is None:
        interp_keys = myda.keys()
    if isinstance(interp_keys, str):
        interp_keys = (interp_keys,)
    old_index = myda.__data__[reg_key]
    new_index = mynumpy.linspace(min(old_index), max(old_index), slices, True)
    result[reg_key] = new_index
    for key in interp_keys:
        if key == reg_key:
            continue
        result[key] = mynumpy.interp(new_index, old_index, myda.__data__[key], method=method)
    return result


def dtype(myda, keys=None):
    if keys is None:
        keys = myda.keys()
    if isinstance(keys, str):
        keys = (keys,)
    result = []
    for key in keys:
        result.append((key, myda.__data__[key].dtype, myda.__data__[key].shape[1:]))
    return numpy.dtype(result)


def from_recarray(a, keys=None, selection=None):
    result = MyDA()
    if keys is None:
        keys = a.dtype.names
    if isinstance(keys, str):
        keys = (keys,)
    if selection:
        a = a[selection]
    for key in keys:
        result[key] = a[key]
    return result


def to_recarray(myda, keys=None, selection=None):
    if keys is None:
        keys = myda.keys()
    if isinstance(keys, str):
        keys = (keys,)
    result = None
    if selection is None:
        selection = numpy.ones(len(myda), dtype='bool')
    for key in keys:
        if result is None:
            first_field = myda.__data__[key][selection]
            result = numpy.empty(len(first_field), dtype=dtype(myda, keys))
            result[key] = first_field
        else:
            result[key] = myda.__data__[key][selection]
    return result


def compact(mydas, index_key, value_key):
    result = MyDA()
    if len(mydas):
        result[index_key] = mydas[0].__data__[index_key]
        result[value_key] = numpy.empty((len(result.__data__[index_key]), len(mydas)),
                                        dtype=mydas[0].__data__[index_key].dtype)
    for i in range(len(mydas)):
        result[value_key][:, i] = mydas[i].__data__[value_key]
    return result


def head(obj, n=5):
    if type(obj) is MyDA:
        print('{')
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
        if len(obj) > n:
            for i in range(n):
                head(obj[i])
                print('...')
        else:
            for i in range(len(obj)):
                head(obj[i])
                print('')
        print(')')
    elif type(obj) is list or type(obj) is numpy.ndarray:
        if len(obj) > n:
            print('[', end=' ')
            for i in range(n):
                head(obj[i])
            print('...]', end=' ')
        else:
            print('[', end=' ')
            for i in range(len(obj)):
                head(obj[i])
            print(']', end=' ')
    else:
        print(repr(obj), end=' ')


class MyDA:
    def __init__(self):
        self.__data__ = OrderedDict()
        self.__length__ = 0

    def __len__(self):
        return self.__length__

    def __getitem__(self, keys):
        if isinstance(keys, str):
            keys = keys
            return self.__data__[keys]
        else:
            return tuple(self[key] for key in keys)

    def __setitem__(self, key, value):
        if not key:
            return
        if not isinstance(key, str):
            raise TypeError('`key` is not string.')
        if len(self.__data__) and len(value) != len(self):
            raise ValueError('new data has not equal length.')
        self.__data__[key] = numpy.array(value).copy()
        if not self.__length__:
            self.__length__ = len(value)

    def __delitem__(self, key):
        self.__data__.__delitem__(key)

    def __repr__(self):
        content = ''
        for key in self.keys():
            content += '{}: {}\n'.format(key, repr(self.__data__[key]))
        return '{\n' + content + '}'

    def __iter__(self):
        return self.__data__.__iter__()

    def values(self):
        return self.__data__.values()

    def keys(self):
        return self.__data__.keys()

    def where(self, selection=None):
        return where(myda=self, selection=selection)

    def orderby(self, keys, reverse=False):
        orderby(myda=self, key=keys, reverse=reverse, result=self)

    def aggregate(self, values=None, funcs=None, groupby=None):
        return aggregate(myda=self, values=values, funcs=funcs, groupby=groupby)

    def apply(self, keys, func):
        apply(myda=self, keys=keys, func=func, result=self)

    def append(self, other):
        return append(myda1=self, myda2=other)

    def regularize(self, reg_key, slices, interp_keys=None, method='linear'):
        return regularize(myda=self, reg_key=reg_key, slices=slices, interp_keys=interp_keys, method=method)

