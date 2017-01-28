# -*- coding: utf-8 -*-

from osgeo import osr, gdal
import numpy
import datetime
import os

__error_stacks_length__ = 'DiffStacksLength'
__error_stacks_length_msg__ = 'The tif stacks\' length are different.'
__error_missing_file__ = 'MissingFile'
__error_missing_file_msg__ = 'The file %s does not exist in the file system.'
__error_time_line_length__ = 'WrongTimeLineLength'
__error_time_line_length_msg__ = 'Time line length (%s) is different of stack time series length (%s).'
__error_wrong_coordinates_format__ = \
    'you must inform a list of coordinates or a string file path to coordinates.'
__error_outbounds__ = 'OutOfBounds'
__error_outbounds_msg__ = 'Coordinate/pixel index out of bounds.'
__error_required_tag__ = 'RequiredTagMissing'
__error_required_tag_msg__ = 'Tag \'%s\' is required but is missing in your file.'
__error_invalid_bbox__ = 'InvalidBoundBox'
__error_invalid_bbox_msg__ = 'Invalid bound box.'
__error_attrs_tags__ = 'BandsTagsError'
__error_attrs_tags_msg__ = 'attrs_paths and attrs_factors tags must have the same length.'


def ord_pair(i, j):
    return numpy.array([i, j])


def to_float(value, decimal):
    result = value
    if decimal != '.':
        result = result.replace(decimal, '.')
    return float(result)


def to_date(value, date_format):
    return datetime.datetime.strptime(value, date_format)


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
            raise Exception(__error_missing_file__, __error_missing_file_msg__ % filename)
        self.srs_wkt = self.dataset.GetProjectionRef()
        self.srs = osr.SpatialReference()
        self.srs.ImportFromWkt(self.srs_wkt)
        self.width = self.dataset.RasterXSize
        self.height = self.dataset.RasterYSize
        self.size = numpy.array([self.width, self.height], dtype=numpy.dtype(int))
        geo_transform = self.dataset.GetGeoTransform()
        self.geo_resolution = ord_pair(geo_transform[Mygdal.GT_X_RES], geo_transform[Mygdal.GT_Y_RES])
        self.geo_skew = ord_pair(geo_transform[Mygdal.GT_Y_SKEW], geo_transform[Mygdal.GT_X_SKEW])
        self.geo_ul = ord_pair(geo_transform[Mygdal.GT_X_UL], geo_transform[Mygdal.GT_Y_UL])
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
            raise Exception(__error_invalid_bbox__, __error_invalid_bbox_msg__)
        if bbox_ul:
            if numpy.any(bbox_ul < (ul * numpy.sign(self.geo_resolution))):
                raise Exception(__error_outbounds__, __error_outbounds_msg__)
            ul = bbox_ul
        if bbox_lr:
            if numpy.any(bbox_lr > (lr * numpy.sign(self.geo_resolution))):
                raise Exception(__error_outbounds__, __error_outbounds_msg__)
            lr = bbox_lr
        return ul + numpy.random.rand(n, 2) * (lr - ul)

    def get_random_pixels(self, n=1, bbox_ul=None, bbox_lr=None, seed=None):
        if seed:
            numpy.random.seed(seed)
        ul = numpy.array([0, 0], dtype=numpy.dtype(int))
        lr = self.size - 1
        if numpy.any(ul > lr):
            raise Exception(__error_invalid_bbox__, __error_invalid_bbox_msg__)
        if bbox_ul:
            if numpy.any(bbox_ul < ul):
                raise Exception(__error_outbounds__, __error_outbounds_msg__)
            ul = bbox_ul
        if bbox_lr:
            if numpy.any(bbox_lr > lr):
                raise Exception(__error_outbounds__, __error_outbounds_msg__)
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
        Verifies if each pixel in pixels belongs to image size.
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
                result[good_data] = numpy.where(result_good_data < min_value, default_value, result_good_data)
            if max_value is not None:
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