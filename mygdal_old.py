# -*- coding: utf-8 -*-

import os
import osr
import gdal
import numpy
import datetime

__error_missing_file_msg__ = 'The file %s does not exist in the file system.'
__error_outbounds_msg__ = 'Coordinate/pixel index out of bounds.'
__error_invalid_bbox_msg__ = 'Invalid bound box.'

gdal.AllRegister()


def ord_pair(i, j):
    return numpy.array([i, j])


def to_float(value, decimal):
    result = value
    if decimal != '.':
        result = result.replace(decimal, '.')
    return float(result)


def to_date(value, date_format):
    return datetime.datetime.strptime(value, date_format)


# coordinates constants indexes
X = 0
Y = 1
dim2D = 2
dim3D = 3

# gdal resample algorithms' constants
GRA_NearestNeighbour = gdal.GRA_NearestNeighbour
GRA_Bilinear = gdal.GRA_Bilinear
GRA_Cubic = gdal.GRA_Cubic
GRA_CubicSpline = gdal.GRA_CubicSpline
GRA_Lanczos = gdal.GRA_Lanczos
GRA_Average = gdal.GRA_Average
GRA_Mode = gdal.GRA_Mode

# gdal data types' constants
GDT_Unknown = gdal.GDT_Unknown
GDT_Byte = gdal.GDT_Byte
GDT_UInt16 = gdal.GDT_UInt16
GDT_Int16 = gdal.GDT_Int16
GDT_UInt32 = gdal.GDT_UInt32
GDT_Int32 = gdal.GDT_Int32
GDT_Float32 = gdal.GDT_Float32
GDT_Float64 = gdal.GDT_Float64
GDT_CInt16 = gdal.GDT_CInt16
GDT_CInt32 = gdal.GDT_CInt32
GDT_CFloat32 = gdal.GDT_CFloat32
GDT_CFloat64 = gdal.GDT_CFloat64
GDT_TypeCount = gdal.GDT_TypeCount

#******************************************************************************
#  GDT_to_dtype function is an adaptation of GDALTypeCodeToNumericTypeCode
#  from gdalnumeric.
#
#  Copyright (c) 2000, Frank Warmerdam
#
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
#
#******************************************************************************

def GDT_to_dtype(gdt):
    if gdt == GDT_Byte:
        return numpy.uint8
    elif gdt == GDT_UInt16:
        return numpy.uint16
    elif gdt == GDT_Int16:
        return numpy.int16
    elif gdt == GDT_UInt32:
        return numpy.uint32
    elif gdt == GDT_Int32:
        return numpy.int32
    elif gdt == GDT_Float32:
        return numpy.float32
    elif gdt == GDT_Float64:
        return numpy.float64
    elif gdt == GDT_CInt16:
        return numpy.complex32
    elif gdt == GDT_CInt32:
        return numpy.complex32
    elif gdt == GDT_CFloat32:
        return numpy.complex32
    elif gdt == GDT_CFloat64:
        return numpy.complex64
    else:
        return None

def get_proj(anyproj):
    srs = osr.SpatialReference()
    try:
        srs.ImportFromEPSG(anyproj)
    except TypeError:
        srs.SetFromUserInput(anyproj)
    if not srs.IsGeographic() and not srs.IsProjected():
        raise ValueError('informed `anyproj` argument could not be resolved.')
    return srs.ExportToWkt()


def reproj(geo, to_proj, geo_reproj=None):
    result = geo.copy()
    result.__reproject__(to_proj=to_proj, srs=geo_reproj)
    return result


def rand_geolocs(bbox, n=1):
    """
    :param bbox: geobbox
    :param n: int
    :return: geolocs
    """
    return numpy.random.rand(n, dim2D) * bbox.size


def __get_srs__(proj):
    result = osr.SpatialReference()
    result.ImportFromWkt(proj)
    if not result.IsGeographic() and not result.IsProjected():
        raise ValueError('informed `any_proj` argument could not be resolved.')
    return result


class geoclass:
    def __init__(self, anyproj):
        self.__proj__ = get_proj(anyproj=anyproj)
        self.__srs__ = __get_srs__(self.__proj__)

    def __get_geo_reproj__(self, to_proj):
        """
        Returns a osr.CoordinateTransformation object to performs geolocs reprojections.
        :param to_proj: str
        :return: osr.CoordinateTransformation
        """
        to_srs = __get_srs__(to_proj)
        return osr.CoordinateTransformation(self.__srs__, to_srs)

    def __points_reproject__(self, pts, from_proj=None, to_proj=None, geo_reproj=None):
        """
        Reprojects all points in @pts from a given system of reference to the image's one.
        Return an array containing the given points.
        :param pts: numpy.ndarray
        :param from_proj: str
        :param to_proj: str
        :param geo_reproj: osr.CoordinateTransformation
        :return: numpy.ndarray
        """
        if from_proj is not None and to_proj is not None and geo_reproj is None:
            geo_reproj = self.__get_geo_reproj__(to_proj)
        elif not (from_proj is None and to_proj is None and geo_reproj is not None):
            raise TypeError('invalid informed arguments.')
        return numpy.array(geo_reproj.TransformPoints(pts))

    def __reproject__(self, to_proj, srs=None):
        """
        :param to_proj: str
        """
        self.__proj__ = to_proj
        self.__srs__ = srs

    def has_same_proj(self, other):
        return self.__srs__.IsSame(other.__srs__)


class geolocs(geoclass):
    def __init__(self, anyproj, x=None, y=None, points=None):
        """
        :param anyproj: str
        :param x: numpy.ndarray
        :param y: numpy.ndarray
        :param points: numpy.ndarray
        """

        srs = __get_srs__()
        super().__init__(anyproj=anyproj)
        if x is not None and y is not None and points is None:
            if hasattr(x, '__len__') and hasattr(y, '__len__'):
                # noinspection PyTypeChecker
                if len(x) != len(y):
                    raise ValueError('arguments `x` and `y` have not the same length.')
                self.__points__ = numpy.array([x, y]).T
            else:
                self.__points__ = numpy.array([[x, y]])
        elif x is None and y is None and points is not None:
            if len(points.shape) == 1 and (points.shape[0] == dim2D or points.shape[0] == dim3D):
                self.__points__ = numpy.array([points[:dim2D]])
            elif len(points.shape) == 2 and (points.shape[1] == dim2D or points.shape[1] == dim3D):
                self.__points__ = numpy.array(points[:, :dim2D])
            else:
                raise ValueError('argument `points` has invalid points.')
        elif x is None and y is None and points is None:
            self.__points__ = None
        else:
            raise TypeError('invalid informed arguments.')

    def __len__(self):
        return len(self.__points__)

    def __getitem__(self, item):
        return geolocs(self.proj, points=self.__points__[item])

    def __iter__(self):
        return iter(self.__points__)

    def __array__(self):
        return self.__points__

    def __add__(self, other):
        """
        :param other: Union[geolocs, numpy.ndarray]
        :return: geolocs
        """
        if isinstance(other, geolocs):
            if not self.has_same_proj(other):
                raise TypeError('inconsistent spatial references.')
        return geolocs(self.proj, points=self.__points__ + other)

    def __sub__(self, other):
        """
        returns an array with the differences between coordinates
        :param other: Union[geolocs, numpy.ndarray]
        :return: geolocs
        """
        if isinstance(other, geolocs):
            if not self.has_same_proj(other):
                raise TypeError('inconsistent spatial references.')
        return geolocs(self.proj, points=self.__points__ - other)

    def __mul__(self, other):
        """
        :param other: numpy.ndarray
        :return: geolocs
        """
        if isinstance(other, geolocs):
            if not self.has_same_proj(other):
                raise TypeError('inconsistent spatial references.')
        return geolocs(self.proj, points=self.__points__ * other)

    def __gt__(self, other):
        """
        :param other: numpy.ndarray
        :return: numpy.ndarray
        """
        if isinstance(other, geolocs):
            if not self.has_same_proj(other):
                raise TypeError('inconsistent spatial references.')
        return self.__points__ > other

    def __ge__(self, other):
        """
        :param other: numpy.ndarray
        :return: numpy.ndarray
        """
        if isinstance(other, geolocs):
            if not self.has_same_proj(other):
                raise TypeError('inconsistent spatial references.')
        return self.__points__ >= other

    def __eq__(self, other):
        """
        :param other: numpy.ndarray
        :return: numpy.ndarray
        """
        if isinstance(other, geolocs):
            if not self.has_same_proj(other):
                raise TypeError('inconsistent spatial references.')
        return self.__points__ == other

    def __le__(self, other):
        """
        :param other: numpy.ndarray
        :return: numpy.ndarray
        """
        if isinstance(other, geolocs):
            if not self.has_same_proj(other):
                raise TypeError('inconsistent spatial references.')
        return self.__points__ <= other

    def __lt__(self, other):
        """
        :param other: numpy.ndarray
        :return: numpy.ndarray
        """
        if isinstance(other, geolocs):
            if not self.has_same_proj(other):
                raise TypeError('inconsistent spatial references.')
        return self.__points__ < other

    def __ne__(self, other):
        """
        :param other: numpy.ndarray
        :return: numpy.ndarray
        """
        if isinstance(other, geolocs):
            if not self.has_same_proj(other):
                raise TypeError('inconsistent spatial references.')
        return self.__points__ != other

    def __reproject__(self, to_proj, srs=None):
        """
        :param to_proj: str
        :param srs: osr.CoordinateTransformation
        """
        if srs is None:
            srs = self.__get_geo_reproj__(to_proj)
        self.__points__ = self.__points_reproject__(pts=self.__points__, geo_reproj=srs)
        if self.__points__.shape[1] == dim3D:
            self.__points__ = self.__points__[:, :dim2D]
        super().__reproject__(to_proj=to_proj, srs=srs)

    @property
    def proj(self):
        return self.__proj__

    @property
    def x(self):
        """
        :return: numpy.ndarray
        """
        if self.__points__ is not None:
            if len(self.__points__) == 1:
                return self.__points__[0][X]
            elif len(self.__points__) > 1:
                return self.__points__[:, X]
        return None

    @property
    def y(self):
        """
        :return: numpy.ndarray
        """
        if self.__points__ is not None:
            if len(self.__points__) == 1:
                return self.__points__[0][Y]
            elif len(self.__points__) > 1:
                return self.__points__[:, Y]
        return None

    @property
    def points(self):
        return self.__points__

    def head(self):
        """
        returns the first geolocs point in self.points
        :return: geolocs
        """
        return geolocs(self.proj, points=self.__points__[0])

    def copy(self):
        return geolocs(self.proj, points=self.__points__)


class geobbox(geoclass):
    def __init__(self, geo_ul=None, geo_lr=None):
        """
        :type geo_ul: geolocs
        :type geo_lr: geolocs
        """
        if not geo_ul.has_same_proj(geo_lr):
            raise TypeError('inconsistent spatial references.')
        # noinspection PyTypeChecker
        if numpy.any((geo_ul * numpy.sign([1, -1])) > (geo_lr * numpy.sign([1, -1]))):
            raise ValueError('inconsistent bounding coordinates.')
        super().__init__(geo_ul.proj)
        self.__ul__ = geo_ul.head()
        self.__lr__ = geo_lr.head()
        self.__ur__ = geolocs(self.__ul__.proj, x=self.__lr__.x, y=self.__ul__.y)
        self.__ll__ = geolocs(self.__ul__.proj, x=self.__ul__.x, y=self.__lr__.y)
        self.__size__ = (self.__ul__ - self.__lr__).points.copy()

    def __reproject__(self, to_proj, srs=None):
        """
        :param to_proj: str
        :param srs: osr.CoordinateTransformation
        """
        if srs is None:
            srs = self.__get_geo_reproj__(to_proj)
        self.__ul__.__reproject__(to_proj=to_proj, srs=srs)
        self.__lr__.__reproject__(to_proj=to_proj, srs=srs)
        self.__ur__.__reproject__(to_proj=to_proj, srs=srs)
        self.__ll__.__reproject__(to_proj=to_proj, srs=srs)
        new_ulx = numpy.min(numpy.array(self.__ul__.x, self.__ll__.x))
        new_uly = numpy.max(numpy.array(self.__ul__.y, self.__ur__.y))
        new_lrx = numpy.max(numpy.array(self.__lr__.x, self.__ur__.x))
        new_lry = numpy.min(numpy.array(self.__ll__.y, self.__lr__.y))
        self.__ul__ = geolocs(self.__ul__.proj, x=new_ulx, y=new_uly)
        self.__lr__ = geolocs(self.__ul__.proj, x=new_lrx, y=new_lry)
        self.__ur__ = geolocs(self.__ul__.proj, x=new_lrx, y=new_uly)
        self.__ll__ = geolocs(self.__ul__.proj, x=new_ulx, y=new_lry)
        self.__size__ = (self.__ul__ - self.__lr__).points.copy()
        super().__reproject__(to_proj=to_proj, srs=srs)

    @property
    def ul(self):
        return self.__ul__

    @property
    def lr(self):
        return self.__lr__

    @property
    def size(self):
        return self.__size__

    @property
    def proj(self):
        return self.__ul__.proj

    def inside(self, geo):
        """
        :param geo: geolocs
        :return: numpy.ndarray
        """
        if self.proj != geo.proj:
            raise TypeError('inconsistent spatial references.')
        if isinstance(geo, geobbox):
            return ((self.__ul__ * numpy.sign([1, -1])) <= (geo.__ul__ * numpy.sign([1, -1]))) * \
                   ((geo.__lr__ * numpy.sign([1, -1])) <= (self.__lr__ * numpy.sign([1, -1])))
        else:
            return ((self.__ul__ * numpy.sign([1, -1])) <= (geo * numpy.sign([1, -1]))) * \
                   ((geo * numpy.sign([1, -1])) <= (self.__lr__ * numpy.sign([1, -1])))

    def copy(self):
        return geobbox(geo_ul=self.ul, geo_lr=self.lr)


# GeoTransform constants indexes
GT_X_UL = 0
GT_X_RES = 1
GT_X_SKEW = 2
GT_Y_UL = 3
GT_Y_SKEW = 4
GT_Y_RES = 5


class Mygdal:
    # GeoTransform Constants Indexes
    GT_X = 0
    GT_Y = 1
    GT_X_UL = 0
    GT_X_RES = 1
    GT_X_SKEW = 2
    GT_Y_UL = 3
    GT_Y_SKEW = 4
    GT_Y_RES = 5

    def __init__(self, filename):
        self.dataset = gdal.Open(filename)
        if not self.dataset:
            raise IOError(__error_missing_file_msg__ % filename)
        self.srs_wkt = self.dataset.GetProjectionRef()
        self.srs = osr.SpatialReference()
        self.srs.ImportFromWkt(self.srs_wkt)
        self.width = self.dataset.RasterXSize
        self.height = self.dataset.RasterYSize
        self.size = numpy.array([self.width, self.height], dtype=numpy.dtype(int))
        geo_reproj = self.dataset.GetGeoTransform()
        self.geo_resolution = ord_pair(geo_reproj[Mygdal.GT_X_RES], geo_reproj[Mygdal.GT_Y_RES])
        self.geo_skew = ord_pair(geo_reproj[Mygdal.GT_Y_SKEW], geo_reproj[Mygdal.GT_X_SKEW])
        self.geo_ul = ord_pair(geo_reproj[Mygdal.GT_X_UL], geo_reproj[Mygdal.GT_Y_UL])
        self.geo_lr = self.pixels_to_geolocs(ord_pair(self.width - 1, self.height - 1))
        self.geo_size = self.geo_lr - self.geo_ul
        self.geo_size_abs = numpy.abs(self.geo_size)
        self.attrs_len = self.dataset.RasterCount
        self.attrs_nodata = numpy.array([self.dataset.GetRasterBand(i).GetNoDataValue()
                                         for i in range(1, self.dataset.RasterCount + 1)])

    def close_dataset(self):
        self.dataset = None

    def get_random_geolocs(self, n=1, bbox_ul=None, bbox_lr=None, seed=None):
        if seed:
            numpy.random.seed(seed)
        ul = self.geo_ul
        lr = self.geo_lr
        if numpy.any((ul * numpy.sign(self.geo_resolution)) > (lr * numpy.sign(self.geo_resolution))):
            raise ValueError(__error_invalid_bbox_msg__)
        if bbox_ul:
            if numpy.any(bbox_ul < (ul * numpy.sign(self.geo_resolution))):
                raise ValueError(__error_outbounds_msg__)
            ul = bbox_ul
        if bbox_lr:
            if numpy.any(bbox_lr > (lr * numpy.sign(self.geo_resolution))):
                raise ValueError(__error_outbounds_msg__)
            lr = bbox_lr
        return ul + numpy.random.rand(n, 2) * (lr - ul)

    def get_random_pixels(self, n=1, bbox_ul=None, bbox_lr=None, seed=None):
        if seed:
            numpy.random.seed(seed)
        ul = numpy.array([0, 0], dtype=numpy.dtype(int))
        lr = self.size - 1
        if numpy.any(ul > lr):
            raise ValueError(__error_invalid_bbox_msg__)
        if bbox_ul:
            if numpy.any(bbox_ul < ul):
                raise ValueError(__error_outbounds_msg__)
            ul = bbox_ul
        if bbox_lr:
            if numpy.any(bbox_lr > lr):
                raise ValueError(__error_outbounds_msg__)
            lr = bbox_lr
        return ul + numpy.array(numpy.random.rand(n, 2) * (lr - ul), dtype=numpy.dtype(int))

    def mask_valid_geolocs(self, geolocs):
        """
        Verifies if each point in geolocs belongs to image bounding box.
        Points in @geolocs must be in same system of reference than image's.
        Returns True if all points are valid, False otherwise.
        :param geolocs: numpy.array
        :return: bool
        """
        return (geolocs >= (self.geo_ul * numpy.sign(self.geo_resolution))) * \
               (geolocs <= (self.geo_lr * numpy.sign(self.geo_resolution)))

    def mask_valid_pixels(self, pixels):
        """
        Verifies if each pixel in pixels belongs to image __size__.
        Pixels in @pixels parameter must be in same system of reference than image's.
        Returns True if all pixels are valid, False otherwise.
        :param pixels: numpy.array
        :return: bool
        """
        return (pixels >= numpy.array([0, 0])) * (pixels <= self.size)

    def pixels_to_geolocs(self, pixels):
        result = numpy.array([])
        if pixels is not None and len(pixels):
            result = self.geo_ul + self.geo_resolution * pixels + self.geo_skew * pixels
        return result

    def geolocs_to_pixels(self, geolocs):
        result = numpy.array([])
        if geolocs is not None and len(geolocs):
            result = (geolocs - self.geo_ul)
            with numpy.errstate(divide='ignore', invalid='ignore'):
                result_geo_resolution = numpy.floor_divide(result, self.geo_resolution)
                result_geo_skew = numpy.floor_divide(result, self.geo_skew)
                result_geo_resolution[~numpy.isfinite(result_geo_resolution)] = 0
                result_geo_skew[~numpy.isfinite(result_geo_skew)] = 0
            result = numpy.array(result_geo_resolution + result_geo_skew, dtype=numpy.dtype(int))
        return result

    def read_pixel(self, pixel, factor_value=1.0, default_value=numpy.nan, min_value=None, max_value=None, dtype='f8'):
        pixel = numpy.asarray(pixel)
        if len(pixel):
            result = self.dataset.ReadAsArray(int(pixel[Mygdal.GT_X]), int(pixel[Mygdal.GT_Y]),
                                              xsize=1, ysize=1)
            result = numpy.reshape(result, self.attrs_len)
            good_data = result != self.attrs_nodata
            result[~good_data] = default_value
            result_good_data = result[good_data] * factor_value
            result[good_data] = result_good_data
            if min_value is not None:
                # noinspection PyTypeChecker
                result[good_data] = numpy.where(result_good_data < min_value, default_value, result_good_data)
            if max_value is not None:
                # noinspection PyTypeChecker
                result[good_data] = numpy.where(result_good_data > max_value, default_value, result_good_data)
            return result.astype(dtype)

    def read_pixels(self, pixels, factor_value=1.0, default_value=None, min_value=None, max_value=None):
        pixels = numpy.asarray(pixels)
        if len(pixels):
            result = self.read_pixel(pixels[0], factor_value, default_value, min_value, max_value)
            for i in range(1, len(pixels)):
                result_pixel = self.read_pixel(pixels[i], factor_value, default_value, min_value, max_value)
                result = numpy.vstack((result, result_pixel))
            return result

    def reproject_geolocs_from(self, geolocs, geo_srs_wkt_from):
        """
        Reprojects all points in @geolocs from a given system of reference to the image's one.
        Return an array containing the given points.
        :param geolocs: numpy.array
        :param geo_srs_wkt_from: string
        :return: numpy.array
        """
        if self.srs_wkt == geo_srs_wkt_from:
            return geolocs
        srs_from = osr.SpatialReference()
        srs_from.ImportFromWkt(geo_srs_wkt_from)
        transform = osr.CoordinateTransformation(srs_from, self.srs)
        result = numpy.array(transform.TransformPoints(geolocs))[:, 0:2]
        return result

    @staticmethod
    def mask_nodata_pixel_attrs(pixel_attrs, values_nodata):
        """
        Masks as False all nodata value found in pixel_attrs. Other values will be masked as True.
        :param pixel_attrs: numpy.array
        :param values_nodata: numpy.array
        :return: numpy.array
        """
        return pixel_attrs != values_nodata

    @staticmethod
    def mask_pixel_attrs_range(pixel_attrs, min_range=None, max_range=None):
        result = pixel_attrs
        if min_range:
            result = result >= min_range
        if max_range:
            result = result <= max_range
        return result