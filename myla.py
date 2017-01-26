# -*- coding: utf-8 -*-

from collections import OrderedDict
import numpy


class MyLA:
    def __init__(self):
        self.__data__ = []
        self.__shape__ = None
        self.__dtype__ = None

    def __len__(self):
        return self.__shape__[0]

    def __getitem__(self, item):
        if isinstance(item, list) or isinstance(item, tuple):
            result = numpy.empty(len(self), dtype=array_dtype(self))
            for key in result.dtype.names:
                result[key] = self.__data__[key]
            return result
        else:
            return self.__data__[item]

    def __setitem__(self, key, value):
        if not isinstance(value, numpy.ndarray):
            value = numpy.asarray(value)
        if value.shape != self.__shape__:
            raise ValueError("New data has not same shape as 'self'.")
        self.__data__[key] = numpy.asarray(value).copy()
        
    def __delitem__(self, key):
        self.__data__.__delitem__(key)

    def __repr__(self):
        content = ''
        for key in self.cols():
            content += '{}: {}\n'.format(key, repr(self[key]))
        return '{\n' + content + '}'

    def __iter__(self):
        return self.__data__.__iter__()
    
    @property
    def count(self):
        return len(self.__data__)
    
    def append(self, value):
        if not isinstance(value, numpy.ndarray):
            value = numpy.asarray(value).copy()
        if self.__shape__ and value.shape != self.__shape__:
            raise ValueError("New data has not same shape as 'self'.")
        self.__data__.append(value.copy())
        if not self.__shape__:
            self.__shape__ = value.shape
            self.__dtype__ = value.dtype

    def cols(self):
        return range(self.count)

    def orderby(self, keys, reverse=False):
        arg_sort = numpy.argsort(self[keys], axis=-1)
        if reverse:
            arg_sort = arg_sort[::-1]
        for key in self:
            self.__data__[key] = self.__data__[key][arg_sort]

    def union(self, other):
        if not isinstance(other, MyLA):
            raise TypeError('Argument parameter is not MyLA.')
        try:
            for key in self:
                self.__data__[key] = numpy.append(self.__data__[key], other.__data__[key], axis=0)
        except KeyError:
            raise ValueError("Argument has not the same keys as 'self'.")

    def to_recarray(self, keys=None, selection=None):
        return to_array(self, keys, selection)


def where(myla, selection):
    result = MyLA()
    for col in myla.cols():
        result.append(myla.__data__[col][selection])
    return result


def orderby(myla, cols, reverse=False):
    result = MyLA()
    arg_sort = numpy.argsort(myla[cols], axis=-1)
    if reverse:
        arg_sort = arg_sort[::-1]
    for cols in myla:
        result.append(myla.__data__[cols][arg_sort])
    return result


def aggregate(myla, value_cols, func, groupby_cols):
    result = MyLA()
    groups, groups_args = numpy.unique(myla[groupby_cols], return_inverse=True)
    if not isinstance(groupby_cols, int):
        groupby_cols = '_'.join(groupby_cols)
    result[groupby_cols] = groups
    if isinstance(value_cols, int):
        result[value_cols] = numpy.array([func(myla.__data__[value_cols][numpy.where(groups_args == i)])
                                          for i in range(len(groups))])
    else:
        for col in value_cols:
            result.append(numpy.array([func(myla.__data__[col][numpy.where(groups_args == i)])
                                       for i in range(len(groups))]))
    return result


def union(myla1, myla2):
    if not (isinstance(myla1, MyLA) and isinstance(myla2, MyLA)):
        raise TypeError('One or both parameters are not MyLA.')
    if myla1.__shape__ != myla2.__shape__:
        raise ValueError('Arguments have not the same shapes.')
    result = MyLA()
    for col in myla1.cols():
        result.append(numpy.append(myla1.__data__[col], myla2.__data__[col], axis=0))
    return result


def from_array(a, cols=None, selection=None):
    if len(a.shape) <= 1:
        raise ValueError('Array has not at least 2 dimensions.')
    result = MyLA()
    if not cols:
        cols = range(len(a))
    if selection:
        a = a[selection]
    for col in cols:
        result.append(a[col])
    return result


def to_array(myla, cols=None, selection=None):
    result_shape = (numpy.count_nonzero(selection) if selection else len(myla),
                    numpy.count_nonzero(cols) if cols else myla.count) + myla.__shape__[2:]
    print(str(result_shape))
    result = numpy.empty(result_shape, dtype=myla.__dtype__)
    if selection:
        for i in range(len(result)):
            result[i] = myla.__data__[i][selection]
    else:
        for i in range(len(result)):
            result[i] = myla.__data__[i]
    return result


def head(obj, n=5):
    if type(obj) is MyLA:
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
