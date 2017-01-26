# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy
import mynumpy


class MyDA:
    def __init__(self):
        self.__data__ = OrderedDict()
        self.__length__ = 0

    def __len__(self):
        return self.__length__

    def __getitem__(self, item):
        if isinstance(item, list) or isinstance(item, tuple):
            result = numpy.empty(len(self), dtype=self.dtype(item))
            for key in result.dtype.names:
                result[key] = self.__data__[key]
            return result
        else:
            return self.__data__[item]

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError('Key is not string.')
        if len(self.__data__) and len(value) != len(self):
            raise ValueError('New data has not equal length.')
        self.__data__[key] = numpy.array(value).copy()
        if not self.__length__:
            self.__length__ = len(value)

    def __delitem__(self, key):
        self.__data__.__delitem__(key)

    def __repr__(self):
        content = ''
        for key in self.keys():
            content += '{}: {}\n'.format(key, repr(self[key]))
        return '{\n' + content + '}'

    def __iter__(self):
        return self.__data__.__iter__()

    def values(self):
        return self.__data__.values()

    def keys(self):
        return self.__data__.keys()

    def where(self, selection):
        return where(self, selection)

    def orderby(self, keys, reverse=False):
        return orderby(self, keys, reverse, self)

    def aggregate(self, value_keys, funcs=None, groupby_keys=None):
        return aggregate(self, value_keys, funcs, groupby_keys)

    def apply(self, func, apply_keys=None):
        return apply(self, func, apply_keys)

    def union(self, other):
        return union(self, other)

    def recarray(self, keys=None, selection=None):
        return recarray(self, keys, selection)

    def dtype(self, keys=None):
        if not keys:
            keys = self.__data__.keys()
        return numpy.dtype([(key, self.__data__[key].dtype, self.__data__[key].shape[1:])
                            for key in keys])


def where(myda, selection, result=None):
    if not result:
        result = MyDA()
    for key in myda.__data__:
        result[key] = myda.__data__[key][selection]
    return result


def orderby(myda, keys, reverse=False, result=None):
    if not result:
        result = MyDA()
    arg_sort = numpy.argsort(myda[keys], axis=-1)
    if reverse:
        arg_sort = arg_sort[::-1]
    for keys in myda:
        result.__data__[keys] = myda.__data__[keys][arg_sort]
    return result


def aggregate(myda, value_keys, funcs=None, groupby_keys=None, result=None):
    if not result:
        result = MyDA()
    if not funcs:
        def funcs(x):
            return x
    if groupby_keys:
        groups, groups_args = numpy.unique(myda[groupby_keys], return_inverse=True)
        if not isinstance(groupby_keys, str):
            groupby_keys = '_'.join(groupby_keys)
        result[groupby_keys] = groups
        if isinstance(value_keys, list) or isinstance(value_keys, tuple):
            if isinstance(funcs, list) or isinstance(funcs, tuple):
                for i, key in enumerate(value_keys):
                    result[key] = numpy.array([funcs[i](myda.__data__[key][numpy.where(groups_args == i)])
                                               for i in range(len(groups))])
            else:
                for key in value_keys:
                    result[key] = numpy.array([funcs(myda.__data__[key][numpy.where(groups_args == i)])
                                               for i in range(len(groups))])
        else:
            result[value_keys] = numpy.array([funcs(myda.__data__[value_keys][numpy.where(groups_args == i)])
                                              for i in range(len(groups))])
    else:
        if isinstance(value_keys, list) or isinstance(value_keys, tuple):
            if isinstance(funcs, list) or isinstance(funcs, tuple):
                for i, key in enumerate(value_keys):
                    result[key] = numpy.array([funcs[i](myda.__data__[key])])
            else:
                for key in value_keys:
                    result[key] = numpy.array([funcs(myda.__data__[key])])
        else:
            result[value_keys] = numpy.array([funcs(myda.__data__[value_keys])])
    return result


def apply(myda, func, apply_keys=None, result=None):
    if not result:
        result = MyDA()
    if not apply_keys:
        apply_keys = myda.keys()
    for key in myda.keys():
        if key in apply_keys:
            result[key] = numpy.array([func(value) for value in myda[key]])
        else:
            result[key] = myda[key]
    return result


def regularize(myda, reg_key, slices, interp_keys=None, method='linear', result=None):
    if not result:
        result = MyDA()
    if not interp_keys:
        interp_keys = myda.keys()
    new_reg = mynumpy.linspace(min(myda.__data__[reg_key]), max(myda.__data__[reg_key]), slices, True)
    result[reg_key] = new_reg
    for key in interp_keys:
        if key == reg_key:
            continue
        result[key] = mynumpy.interp(reg_key, myda.__data__[reg_key], myda.__data__[key])
    return result


def union(myda1, myda2, result=None):
    if not (isinstance(myda1, MyDA) and isinstance(myda2, MyDA)):
        raise TypeError('One or both parameters are not MyDA.')
    if not result:
        result = MyDA()
    try:
        for key in myda1:
            result[key] = numpy.append(myda1.__data__[key], myda2.__data__[key], axis=0)
    except KeyError:
        raise ValueError('Arguments have not the same keys.')
    return result


def from_recarray(a, keys=None, selection=None):
    result = MyDA()
    if not keys:
        keys = a.dtype.names
    if selection:
        a = a[selection]
    for key in keys:
        result[key] = a[key]
    return result


def recarray(myda, keys=None, selection=None):
    if not keys:
        keys = myda.keys()
    result_len = numpy.count_nonzero(selection) if selection else len(myda)
    result = numpy.empty(result_len, dtype=myda.dtype(keys))
    if selection:
        for key in keys:
            result[key] = myda.__data__[key][selection]
    else:
        for key in keys:
            result[key] = myda.__data__[key]
    return result


def head(obj, n=5):
    if type(obj) is MyDA:
        print('{')
        for key in obj.cols():
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
