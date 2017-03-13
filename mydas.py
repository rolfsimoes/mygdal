# -*- coding: utf-8 -*-

from collections import OrderedDict, defaultdict
import numpy
import mynumpy


def __proc_func_key_param__(**kwargs):
    new_key = []
    func = []
    key = []
    func_args = []
    for k, value in kwargs.items():
        if len(value) != 2 and len(value) != 3:
            raise TypeError('argument tuple (`agg_func`, `key`) not informed.')
        new_key.append(k)
        func.append(value[0])
        key.append(value[1])
        func_args.append(value[2] if len(value) == 3 else None)
    return new_key, func, key, func_args


def select(m, key=..., but=None):
    result = mydas()
    if isinstance(but, str) or but == Ellipsis or but is None:
        but = [but]
    if isinstance(key, str):
        key = key
        result[key] = m.__data__[key]
    elif isinstance(key, tuple) or isinstance(key, list):
        for k in key:
            if k not in but:
                result[k] = m.__data__[k]
    elif isinstance(key, Ellipsis):
        for k in m.keys():
            if k not in but:
                result[k] = m.__data__[k]
    else:
        raise TypeError('key is not `str`, `int`, `slice`, `numpy.ndarray`, `tuple`, `list`, or `Ellipsis`.')
    return result


def where(m, selection):
    result = mydas()
    for k in m.keys():
        result[k] = m.__data__[k][selection]
    return result


def order_by(m, key, reverse=False, result=None):
    if result is None:
        result = mydas()
    if isinstance(key, str):
        key = [key]
    arg_sort = numpy.argsort(m.combine(*key), axis=-1)
    if reverse:
        arg_sort = arg_sort[::-1]
    # invalidate indexes
    result.__index_of__.clear()
    for k in m.keys():
        result[k] = m.__data__[k][arg_sort]
    return result


def group_by(m, key):
    result = m
    if isinstance(key, str):
        key = [key]
    groups = numpy.unique(m.combine(*key))
    result.__groups__ = groups
    result.__groups_key__ = key
    return result


def aggregate(m, **kwargs):
    new_key, agg_func, key, func_args = __proc_func_key_param__(**kwargs)
    if m.__groups_key__:
        groups, groups_args = numpy.unique(m.combine(*m.__groups_key__), return_inverse=True)
        result = mydas()
        for k in m.__groups_key__:
            result[k] = groups[k]
        for i in range(len(key)):
            result[new_key[i]] = numpy.array([agg_func[i](m.__data__[key[i]][numpy.where(groups_args == j)])
                                             for j in range(len(groups))])
    else:
        result = mydas()
        for i in range(len(key)):
            result[new_key[i]] = numpy.array([agg_func[i](m.__data__[key[i]])])
    return result


def apply(m, **kwargs):
    new_key, app_func, key, func_args = __proc_func_key_param__(**kwargs)
    result = mydas()
    for i in range(len(key)):
        result[new_key[i]] = numpy.array([app_func[i](value) for value in m.__data__[key[i]]])
    return result


def append(m1, m2):
    if not (isinstance(m1, mydas) and isinstance(m2, mydas)):
        raise TypeError('one or both parameters are not mydas.')
    result = mydas()
    if len(m1):
        try:
            for k in m1.keys():
                result[k] = numpy.append(m1.__data__[k], m2.__data__[k], axis=0)
        except KeyError:
            raise ValueError('arguments have not same keys.')
    else:
        for k in m2.keys():
            result[k] = m2.__data__[k]
    return result


def bind(m1, m2, m1_key_suffix='_1', m2_key_suffix='_2', result=None):
    if result is None:
        result = mydas()
    for k in m1.keys():
        result['{0}{1}'.format(k, m1_key_suffix)] = m1.__data__[k]
    for k in m2.keys():
        result['{0}{1}'.format(k, m2_key_suffix)] = m2.__data__[k]
    return result


def join(m1, m2, by, m1_key_suffix='_1', m2_key_suffix='_2'):
    result = mydas()
    for i, v in enumerate(m1[by]):
        m2_select = m2.where(selection=m2.index_of(key=by, value=v))
        m1_select = m1.where(selection=[i] * rows(m2_select))
        if rows(m2_select):
            result = result.append(bind(m1_select, m2_select, m1_key_suffix=m1_key_suffix, m2_key_suffix=m2_key_suffix))
    return result


def left_join(m1, m2, by, m1_key_suffix='_1', m2_key_suffix='_2'):
    pass


def right_join(m1, m2, by, m1_key_suffix='_1', m2_key_suffix='_2'):
    pass


def slicer(m, key, slices, interp_keys=None, method='linear'):
    result = mydas()
    if interp_keys is None:
        interp_keys = m.keys()
    if isinstance(interp_keys, str):
        interp_keys = (interp_keys,)
    old_index = m.__data__[key]
    new_index = mynumpy.linspace(min(old_index), max(old_index), slices, True)
    result[key] = new_index
    for i, k in enumerate(interp_keys):
        if k == key:
            continue
        result[k] = mynumpy.interp(new_index, old_index, m.__data__[k], method=method)
    return result


def dtype(m, key=None):
    if key is None:
        key = m.keys()
    if isinstance(key, str):
        key = [key]
    return numpy.dtype([(k, m.__data__[k].dtype, m.__data__[k].shape[1:]) for k in key])


def compact(m, index_key, value_key):
    result = mydas()
    if isinstance(m, mydas):
        if isinstance(value_key, str):
            result[index_key] = m.__data__[index_key]
            result[value_key] = m.__data__[value_key]
        elif value_key:
            result[index_key] = m.__data__[index_key]
            result[value_key[0]] = numpy.empty((rows(m), rows(value_key)),
                                               dtype=m.__data__[value_key[0]].dtype)
            for i in range(rows(value_key)):
                result[value_key[0]][:, i] = m.__data__[value_key[i]]
    elif m:
        if isinstance(value_key, str):
            result[index_key] = m[0].__data__[index_key]
            result[value_key] = numpy.empty((rows(m[0]), rows(m)),
                                            dtype=m[0].__data__[value_key].dtype)
            for i in range(rows(m)):
                result[value_key][:, i] = m[i].__data__[value_key]
        else:
            raise TypeError('inconsistent arguments informed.')
    return result


def from_recarray(a, key=None, selection=None):
    result = mydas()
    if key is None:
        key = a.dtype.names
    if isinstance(key, str):
        key = [key]
    if selection:
        a = a[selection]
    for k in key:
        result[k] = a[k].copy()
    return result


def to_recarray(m, key=None):
    if key is None:
        key = m.keys()
    if isinstance(key, str):
        key = [key]
    result = numpy.empty(rows(m), dtype=dtype(m, key))
    for k in key:
        result[k][:] = m.__data__[k]
    return result


def rows(m):
    if isinstance(m, mydas):
        return m.__rows__
    return len(m)


def head(obj, n=5):
    if type(obj) is mydas:
        print('{')
        if rows(obj) > n:
            for k in obj.keys()[:n]:
                print('{}: '.format(k), end='')
                head(obj.__data__[k])
                print('...')
        else:
            for k in obj.keys():
                print('{}: '.format(k), end='')
                head(obj.__data__[k])
                print('')
        print('}')
    elif type(obj) is str:
        print(repr(obj), end=' ')
    elif type(obj) is numpy.datetime64:
        print("'{}'".format(obj), end=' ')
    elif type(obj) is tuple:
        print('(')
        if rows(obj) > n:
            for i in range(n):
                head(obj[i])
                print('...')
        else:
            for i in range(rows(obj)):
                head(obj[i])
                print('')
        print(')')
    elif type(obj) is list or type(obj) is numpy.ndarray:
        if rows(obj) > n:
            print('[', end=' ')
            for i in range(n):
                head(obj[i])
            print('...]', end=' ')
        else:
            print('[', end=' ')
            for i in range(rows(obj)):
                head(obj[i])
            print(']', end=' ')
    else:
        print(repr(obj), end=' ')


class mydas:
    def __init__(self, data=None):
        self.__rows__ = 0
        self.__data__ = OrderedDict()
        if data is not None:
            for k in data.keys():
                self[k] = data[k]
        self.__groups_len__ = 0
        self.__groups_key__ = []
        self.__index_of__ = defaultdict(lambda: defaultdict(list))

    def __len__(self):
        return len(self.__data__)

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.__data__[index]
        elif isinstance(index, tuple) or isinstance(index, list):
            return (self.__data__[k] for k in index)
        elif isinstance(index, Ellipsis):
            return (self.__data__[k] for k in self.keys())
        else:
            raise TypeError('index is not `str`, `tuple`, `list`, or `Ellipsis`.')

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
        if len(self) and len(value) != rows(self):
            raise ValueError('new data has not same length.')
        if not len(self):
            self.__rows__ = len(value)
        self.__data__[key] = numpy.array(value)

    def __delitem__(self, key):
        self.__data__.__delitem__(key)

    def __repr__(self):
        content = ''
        for k in self.keys():
            content += '{}: {}\n'.format(k, repr(self.__data__[k]))
        return '{\n' + content + '}'

    def __iter__(self):
        return self.__data__.values()

    @property
    def dtype(self):
        return dtype(m=self, key=self.keys())

    @property
    def rows(self):
        return self.__rows__

    def keys(self):
        return tuple(self.__data__.keys())

    def combine(self, *key):
        return to_recarray(m=self, key=key)

    def index_of(self, key, value):
        if not isinstance(key, str):
            raise TypeError('`key` is not string.')
        if key not in self.__index_of__.keys():
            for i, v in enumerate(self.__data__[key]):
                self.__index_of__[key][v].append(i)
        return self.__index_of__[key][value]

    def copy(self):
        result = mydas()
        for k in self.keys():
            result[k] = self.__data__[k]
        result.__groups_len__ = self.__groups_len__
        result.__groups_key__ = self.__groups_key__.copy()
        return result

    def select(self, *key):
        return select(self, key=key)

    def where(self, selection=None):
        return where(m=self, selection=selection)

    def order_by(self, *key, reverse=False):
        return order_by(m=self, key=key, reverse=reverse)

    def group_by(self, *key):
        return group_by(m=self, key=key)

    def aggregate(self, **kwargs):
        return aggregate(m=self, **kwargs)

    def apply(self, **kwargs):
        return apply(m=self, **kwargs)

    def append(self, other):
        return append(m1=self, m2=other)

    def bind(self, other, key_suffix='_1', other_key_suffix='_2'):
        return bind(m1=self, m2=other, m1_key_suffix=key_suffix, m2_key_suffix=other_key_suffix)

    def join(self, other, by, key_suffix='_1', other_key_suffix='_2'):
        return join(m1=self, m2=other, by=by, m1_key_suffix=key_suffix, m2_key_suffix=other_key_suffix)

    def left_join(self, other, coerce_same_keys=False):
        pass

    def right_join(self, other, coerce_same_keys=False):
        pass

    def slicer(self, key, slices, *interp_key, method='linear'):
        return slicer(m=self, key=key, slices=slices, interp_keys=interp_key, method=method)
