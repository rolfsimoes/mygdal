# -*- coding: utf-8 -*-

import numpy
from numpy.linalg.linalg import LinAlgError
from numpy.linalg import inv
from shapely import affinity, geometry
from osgeo import gdal, gdal_array, osr
import os

# gdal open mode
GA_ReadOnly = gdal.GA_ReadOnly
GA_Update = gdal.GA_Update

# default values
DEFAULT_PROJ = 4326

# gdal reference system types
GRS_Wkt = 0
GRS_Proj4 = 1
GRS_Pretty_Wkt = 2
GRS_XML = 3
GRS_USGS = 4

# gdal raster io resample algorithms' constants
GRIORA_NearestNeighbour = gdal.GRIORA_NearestNeighbour
GRIORA_Bilinear = gdal.GRIORA_Bilinear
GRIORA_Cubic = gdal.GRIORA_Cubic
GRIORA_CubicSpline = gdal.GRIORA_CubicSpline
GRIORA_Lanczos = gdal.GRIORA_Lanczos
GRIORA_Average = gdal.GRIORA_Average
GRIORA_Mode = gdal.GRIORA_Mode
GRIORA_Gauss = gdal.GRIORA_Gauss


def _open_raster(raster, mode=GA_ReadOnly):
    result = raster
    if isinstance(raster, str):
        result = gdal.Open(raster, mode)
        if not result:
            raise IOError('file `{0}` is invalid or missing.'.format(raster))
    if not isinstance(result, gdal.Dataset):
        raise ValueError('informed `raster` argument is not a file name nor a valid gdal raster object.')
    return result


def cast_proj(anyproj, return_type=GRS_Proj4):
    srs = osr.SpatialReference()
    if isinstance(anyproj, int):
        srs.ImportFromEPSG(anyproj)
    else:
        srs.SetFromUserInput(anyproj)
    if not srs.IsGeographic() and not srs.IsProjected():
        raise ValueError('unable to resolve coordinate reference system.')
    if return_type == GRS_Wkt:
        return srs.ExportToWkt()
    elif return_type == GRS_Proj4:
        return srs.ExportToProj4()
    elif return_type == GRS_Pretty_Wkt:
        return srs.ExportToPrettyWkt()
    elif return_type == GRS_Pretty_Wkt:
        return srs.ExportToXML()
    elif return_type == GRS_USGS:
        return srs.ExportToUSGS()
    else:
        return srs.ExportToProj4()


def empty(width, height, bands, geotransform, proj_origin=None, nodata=None, dtype=None, **metadata):
    if proj_origin is None:
        proj_origin = cast_proj(DEFAULT_PROJ, GRS_Wkt)
    else:
        proj_origin = cast_proj(proj_origin, GRS_Wkt)
    if dtype is None:
        dtype = 'f8'
    array = numpy.zeros(shape=(height, width, bands), dtype=dtype)
    if nodata is not None:
        array[:] = numpy.dtype(dtype).type(nodata)
    return GeoArray(array, geotransform, proj_origin=proj_origin, nodata=nodata, **metadata)


def read(raster, geom=None, band_list=None, dtype=None, resample_alg=GRIORA_NearestNeighbour):
    raster = _open_raster(raster)
    if not isinstance(raster, gdal.Dataset):
        raise ValueError('invalid informed `raster` argument.')
    if raster.RasterCount < 1:
        raise ValueError('informed `raster` has no bands')

    # calculate the interested area
    geotransform = GeoTransform(*raster.GetGeoTransform())
    if geom is None:
        x_off, y_off, x_size, y_size = 0, 0, raster.RasterXSize, raster.RasterYSize
    else:
        geom_raster = geotransform.transform(geometry.box(0, 0, raster.RasterXSize, raster.RasterYSize))
        intersection = geom_raster.intersection(geom)
        if intersection.is_empty:
            raise ValueError('`geom` do not intersects with `raster`.')

        # get offset to read from raster
        x_off, y_off, x_size, y_size = tuple(map(int, geotransform.get_inverse().transform(intersection).bounds))
        x_size, y_size = x_size - x_off, y_size - y_off

        # update resulting geotransform
        x_ul, y_min, _, y_max = intersection.exterior.bounds
        y_ul = y_max if geotransform.y_res < 0 else y_min

        geotransform = GeoTransform(x_ul, geotransform.x_res, geotransform.x_skew,
                                    y_ul, geotransform.y_skew, geotransform.y_res)
    bands = raster.RasterCount if band_list is None else len(band_list)

    # prepare raster buffer: get buffer strides
    if dtype is None:
        dtype = gdal_array.flip_code(raster.GetRasterBand(1).DataType)
    sizeof_dtype = numpy.dtype(dtype).itemsize
    line_space, pixel_space, band_space = sizeof_dtype * x_size * bands, sizeof_dtype * bands, sizeof_dtype

    # read to buffer
    buffer = raster.ReadRaster(xoff=x_off, yoff=y_off, xsize=x_size, ysize=y_size,
                               buf_type=gdal_array.flip_code(numpy.dtype(dtype)),
                               buf_xsize=x_size, buf_ysize=y_size, band_list=band_list,
                               buf_pixel_space=pixel_space, buf_line_space=line_space,
                               buf_band_space=band_space, resample_alg=resample_alg)
    array = numpy.fromstring(buffer, dtype=dtype)
    array.shape = (y_size, x_size, -1)
    return GeoArray(array, geotransform, proj_origin=raster.GetProjection(), **raster.GetMetadata_Dict())


def write(geo_array, file_name, dtype=None, nodata=None, proj=None, gdal_driver='GTiff', overwrite=False):
    if not overwrite and os.path.isfile(file_name):
        raise IOError('file `{}` already exists.'.format(file_name))
    if dtype is None:
        dtype = geo_array.array.dtype
    raster = gdal.GetDriverByName(gdal_driver).Create(utf8_path=file_name,
                                                      xsize=geo_array.width, ysize=geo_array.height,
                                                      bands=geo_array.bands,
                                                      eType=gdal_array.flip_code(numpy.dtype(dtype)))
    if not raster:
        raise IOError('an error occurred while creating gdal dataset.')

    # resolve projection
    if proj is None:
        proj = geo_array.proj_origin
    if proj is None:
        proj = DEFAULT_PROJ
    raster.SetProjection(proj)
    raster.SetGeoTransform(geo_array.geotransform.to_list())

    # convert buffer dtype to informed dtype parameter as to write data to raster.
    buffer = geo_array.array.astype(dtype, copy=True)
    byte_string = buffer.tostring()

    # get buffer strides in order to finda data elements position in `byte_string`
    height_size, width_size, bands_size = buffer.strides
    raster.WriteRaster(xoff=0, yoff=0, xsize=geo_array.width, ysize=geo_array.height,
                       buf_string=byte_string,
                       buf_xsize=geo_array.width, buf_ysize=geo_array.height,
                       buf_type=gdal_array.flip_code(dtype),
                       buf_pixel_space=width_size, buf_line_space=height_size, buf_band_space=bands_size)
    # 2-do: capture gdal errors and treat them appropriately.
    raster.FlushCache()

    if nodata is None:
        nodata = geo_array.nodata
    if nodata is not None:
        for i in range(1, raster.RasterCount + 1):
            raster.GetRasterBand(i).SetNoDataValue(nodata)
    for i in range(1, raster.RasterCount + 1):
        raster.GetRasterBand(i).ComputeStatistics(False)
    raster.SetMetadata(geo_array.metadata)


def raster_box(raster):
    raster = _open_raster(raster)
    if not isinstance(raster, gdal.Dataset):
        raise ValueError('invalid informed `raster` argument.')
    geotransform = GeoTransform(*raster.GetGeoTransform())
    return geotransform.transform(geometry.box(0, 0, raster.RasterXSize, raster.RasterYSize))


class GeoTransform:
    def __init__(self, x_ul, x_res, x_skew, y_ul, y_skew, y_res):
        self.x_ul = x_ul
        self.x_res = x_res
        self.x_skew = x_skew
        self.y_ul = y_ul
        self.y_skew = y_skew
        self.y_res = y_res

    def transform(self, geom):
        return affinity.affine_transform(geom, (self.x_res, self.x_skew, self.y_skew,
                                                self.y_res, self.x_ul, self.y_ul))

    def get_inverse(self):
        try:
            inv_matrix = inv([[self.x_res, self.x_skew], [self.y_skew, self.y_res]])
        except LinAlgError:
            raise ValueError('`geotransform` is not invertible.')
        px_res, px_skew, py_skew, py_res = inv_matrix.ravel()
        px_off, py_off = numpy.matmul(inv_matrix, (self.x_ul, self.y_ul))
        return GeoTransform(-px_off, px_res, px_skew, -py_off, py_skew, py_res)

    def to_list(self):
        return [self.x_ul, self.x_res, self.x_skew, self.y_ul, self.y_skew, self.y_res]

    def to_points(self, px, py):
        pixels = numpy.array([px, py]).T
        return [self.transform(geometry.Point(*pixel)) for pixel in pixels]

    def to_linestring(self, px, py):
        pixels = numpy.array([px, py]).T
        return self.transform(geometry.LineString(pixels))

    def to_linearring(self, px, py):
        pixels = numpy.array([px, py]).T
        return self.transform(geometry.LinearRing(pixels))

    def to_polygon(self, px, py):
        pixels = numpy.array([px, py]).T
        return self.transform(geometry.Polygon(pixels))


class GeoArray:
    def __init__(self, a, geotransform, copy=False, proj_origin=None, nodata=None, **metadata):
        if a.ndim not in (2, 3):
            raise ValueError('`array` has not 2 or 3-dimensions.')
        if a.ndim == 2: a.shape += (-1,)
        self.array = a.copy() if copy else a
        if not isinstance(geotransform, GeoTransform):
            geotransform = GeoTransform(*geotransform)
        self.geotransform = geotransform
        self.inv_geotransform = geotransform.get_inverse()
        self.proj_origin = proj_origin
        self.nodata = nodata
        self.metadata = metadata

    def __getitem__(self, index):
        if isinstance(index, tuple) and len(index) in (1, 2):
            if len(index) == 1:
                index += (slice(None, None),)
            if isinstance(index[0], GeoArray) or isinstance(index[0], geometry.base.BaseGeometry):
                return self.array[self.index(*index)]
            elif isinstance(index[0], geometry.base.BaseMultipartGeometry):
                return [self[geom, index[1]] for geom in index[0].geoms]
        elif isinstance(index, GeoArray) or isinstance(index, geometry.base.BaseGeometry):
            return self.array[self.index(index)]
        elif isinstance(index, geometry.base.BaseMultipartGeometry):
            return tuple(map(self.__getitem__, index.geoms))
        return self.array[index]

    def __setitem__(self, index, value):
        if isinstance(index, tuple) and len(index) in (1, 2):
            if len(index) == 1:
                index += (slice(None, None),)
            if isinstance(index[0], GeoArray) or isinstance(index[0], geometry.base.BaseGeometry):
                self._setdata(index[0], index[1], value)
            elif isinstance(index[0], geometry.base.BaseMultipartGeometry):
                for geom in index[0].geoms:
                    self._setdata(geom, index[1], value)
        elif isinstance(index, GeoArray) or isinstance(index, geometry.base.BaseGeometry):
            self._setdata(index, slice(None, None), value)
        elif isinstance(index, geometry.base.BaseMultipartGeometry):
            for geom in index.geoms:
                self._setdata(geom, slice(None, None), value)
        else:
            self.array[index] = value

    def __array__(self, dtype=None):
        if dtype is None:
            return self.array
        else:
            return self.array.view(dtype)

    def _setdata(self, geom_index, band, value):
        to_array = self.array[self.index(geom_index, band)]
        if self.nodata is None:
            to_array[:] = value
        else:
            to_array[:] = numpy.where(to_array == self.nodata, value, to_array)

    def _index_point(self, geom_index):
        if geom_index.is_empty:
            raise IndexError('informed geo-index is empty.')
        point = self.inv_geotransform.transform(geom_index)
        return int(point.y), int(point.x)

    def _index_linestring(self, geom_index):
        if geom_index.is_empty:
            raise IndexError('informed geo-index is empty.')
        linestring = self.inv_geotransform.transform(geom_index)
        # the slicing `~[::-1]` is used to invert (x, y) to (y, x) as required for index.
        return tuple(numpy.array(linestring.coords, dtype='i4').T[::-1])

    def _index_polygon(self, geom_index):
        if geom_index.is_empty:
            raise IndexError('informed geo-index is empty.')
        polygon = self.inv_geotransform.transform(geom_index)
        px_min, py_min, px_max, py_max = tuple(map(int, polygon.bounds))
        # the `+1` bellow is needed if we want include the last pixel.
        return slice(py_min, py_max + 1), slice(px_min, px_max + 1)

    def index(self, geom_index, band_index=None):
        if band_index is None: band_index = slice(None, None)
        if isinstance(geom_index, GeoArray):
            return self._index_polygon(self.box.intersection(geom_index.box)) + (band_index,)
        if isinstance(geom_index, geometry.Point):
            return self._index_point(geom_index) + (band_index,)
        if isinstance(geom_index, geometry.LineString) or isinstance(geom_index, geometry.LinearRing):
            return self._index_linestring(geom_index) + (band_index,)
        if isinstance(geom_index, geometry.Polygon):
            return self._index_polygon(geom_index) + (band_index,)

    def stack_bands(self, a):
        self.array = numpy.dstack((self.array, a))

    @property
    def height(self):
        return self.array.shape[0]

    @property
    def width(self):
        return self.array.shape[1]

    @property
    def bands(self):
        return self.array.shape[2]

    @property
    def box(self):
        return self.geotransform.transform(geometry.box(0, 0, self.width, self.height))



