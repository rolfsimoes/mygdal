from osgeo import gdal
from osgeo import osr
from osgeo import gdal_array
import numpy
import os

# default values
DEFAULT_PROJ = 4326
DEFAULT_GEOTRANSFORM = (-180.0, 3.6, 0.0, 90.0, 0.0, -3.6)
DEFAULT_SHAPE = (50, 100, 1)

# gdal reference system types
GRS_Wkt = 0
GRS_Proj4 = 1
GRS_Pretty_Wkt = 2
GRS_XML = 3
GRS_USGS = 4

# gdal open mode
GA_ReadOnly = gdal.GA_ReadOnly
GA_Update = gdal.GA_Update

# gdal raster io resample algorithms' constants
GRIORA_NearestNeighbour = gdal.GRIORA_NearestNeighbour
GRIORA_Bilinear = gdal.GRIORA_Bilinear
GRIORA_Cubic = gdal.GRIORA_Cubic
GRIORA_CubicSpline = gdal.GRIORA_CubicSpline
GRIORA_Lanczos = gdal.GRIORA_Lanczos
GRIORA_Average = gdal.GRIORA_Average
GRIORA_Mode = gdal.GRIORA_Mode
GRIORA_Gauss = gdal.GRIORA_Gauss


def _gdal_dtype(dtype):
    return gdal_array.flip_code(numpy.dtype(dtype))


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


def get_crs(anyproj):
    result = osr.SpatialReference()
    if isinstance(anyproj, int):
        result.ImportFromEPSG(anyproj)
    else:
        result.SetFromUserInput(anyproj)
    if not result.IsGeographic() and not result.IsProjected():
        raise ValueError('unable to resolve coordinate reference system.')
    return result


def create_raster(file_name, width, height, bands=1, dtype='u2', gdal_driver='GTiff'):
    result = gdal.GetDriverByName(gdal_driver).Create(utf8_path=file_name, xsize=width, ysize=height,
                                                      bands=bands, eType=_gdal_dtype(dtype))
    if not result:
        raise IOError('an error occurred while creating gdal dataset.')
    return result


def copy_raster(raster, file_name, overwrite=False, compute_statistics=True, gdal_format='GTiff'):
    raster = open_raster(raster)
    if not overwrite and os.path.isfile(file_name):
        raise IOError('file `{}` already exists.'.format(file_name))
    driver = gdal.GetDriverByName(gdal_format)
    metadata = driver.GetMetadata()
    if not (gdal.DCAP_CREATECOPY in metadata and metadata[gdal.DCAP_CREATECOPY] == 'YES'):
        raise IOError('format `{}` does not support copy.'.format(gdal_format))
    result = driver.CreateCopy(file_name, raster, False)
    if not result:
        raise IOError('an error occurred while creating gdal dataset.')
    result.FlushCache()
    return result


def open_raster(raster, mode=GA_ReadOnly):
    result = raster
    if isinstance(raster, str):
        result = gdal.Open(raster, mode)
        if not result:
            raise IOError('file `{0}` is invalid or missing.'.format(raster))
    if not result:
        raise IOError('an error occurred while opening gdal dataset.')
    return result


def empty_geoframe_from_raster(raster, geoframe_shape=None):
    raster = open_raster(raster)
    if geoframe_shape is None:
        geoframe_shape = raster.RasterYSize, raster.RasterXSize
    return GeoFrame(crs=get_crs(raster.GetProjection()), geotransform=raster.GetGeoTransform(),
                    shape=geoframe_shape)


class Window:
    def __init__(self, x_off, y_off, x_size, y_size):
        self.px = x_off
        self.py = y_off
        self.px_size = x_size
        self.py_size = y_size

    def __call__(self):
        return self.px, self.py, self.px_size, self.py_size

    def is_valid(self):
        return self.px >= 0 and self.py >= 0 and self.px_size >= 0 and self.py_size >= 0

    def index(self):
        return slice(self.py, self.py + self.py_size), slice(self.px, self.px + self.px_size)


class GeoFrame:
    def __init__(self, crs=None, geotransform=None, shape=None):
        if crs is None:
            crs = get_crs(DEFAULT_PROJ)
        else:
            if not isinstance(crs, osr.SpatialReference):
                raise ValueError('invalid coordinate reference system.')
        self.crs = crs
        x_res_factor = 1.0
        y_res_factor = 1.0
        if geotransform is None:
            geotransform = DEFAULT_GEOTRANSFORM
            if shape is not None:
                x_res_factor = DEFAULT_SHAPE[1] / shape[1]
                y_res_factor = DEFAULT_SHAPE[0] / shape[0]
        x_ul, x_res, x_skew, y_ul, y_skew, y_res = geotransform
        self.x_ul = x_ul
        self.x_res = x_res * x_res_factor
        self.x_skew = x_skew
        self.y_ul = y_ul
        self.y_skew = y_skew
        self.y_res = y_res * y_res_factor
        if shape is None:
            shape = DEFAULT_SHAPE
        if len(shape) == 2:
            bands = 1
            height, width = shape
        else:
            height, width, bands = shape
        self._width = width
        self._height = height
        self._bands = bands
        self._buffer = None

    def __eq__(self, other):
        """
        Implements `==` operator meaning `same bounds`.
        If geoframe `A` is equal to `B` then it is true that
        `A` is inside `B` and `B` is inside `A`.
        This is expressed as `A == B`.
        :param other: GeoFrame
        :return: bool
        """
        return (self in other) and (other in self)

    def __contains__(self, other):
        """
        Implements `in` operator meaning `contains bounds`.
        If geoframe `B` is inside `A`, then `A` contains `B`.
        This is expressed as `B in A`.
        :param other: GeoFrame
        :return: bool
        """
        try:
            intersection = other * self
        except ValueError:
            return False
        if intersection is None:
            return False
        return (intersection.x_ul == other.x_ul and
                intersection.y_ul == other.y_ul and
                intersection.shape == other.shape)

    def __mul__(self, other):
        """
        Performs an intersection operation between two geoframes.
        Returned resolution and shape are taken from left-side geoframe.
        This is expressed as `A * B`.
        :param other: GeoFrame
        :return: GeoFrame, None
        """
        if other is None:
            return self
        self._verify_crs(other)
        x_ul = max(self.x_ul, other.x_ul)
        x_lr = min(self.x_lr, other.x_lr)
        if x_ul > x_lr:
            return None
        if self.y_lr - self.y_ul < 0:
            y_ul = min(self.y_ul, other.y_ul)
            y_lr = max(self.y_lr, other.y_lr)
            if y_ul < y_lr:
                return None
        else:
            y_ul = max(self.y_ul, other.y_ul)
            y_lr = min(self.y_lr, other.y_lr)
            if y_ul > y_lr:
                return None
        px_ul, py_ul = gdal.ApplyGeoTransform(self.inv_geotransform, x_ul, y_ul)
        px_lr, py_lr = gdal.ApplyGeoTransform(self.inv_geotransform, x_lr, y_lr)
        px_size, py_size = int(px_lr - px_ul), int(py_lr - py_ul)
        if px_size < 1 or py_size < 1:
            return None
        _, x_res, x_skew, _, y_skew, y_res = self.geotransform
        return GeoFrame(crs=self.crs, geotransform=(x_ul, x_res, x_skew, y_ul, y_skew, y_res),
                        shape=(py_size, px_size))

    def __add__(self, other):
        """
        Performs an union operation between two geoframes.
        Returned resolution and shape are taken from left-side geoframe.
        This is expressed as `A + B`.
        :param other: GeoFrame
        :return: GeoFrame
        """
        if other is None:
            return self
        self._verify_crs(other)
        x_ul = min(self.x_ul, other.x_ul)
        x_lr = max(self.x_lr, other.x_lr)
        if self.y_lr - self.y_ul < 0:
            y_ul = max(self.y_ul, other.y_ul)
            y_lr = min(self.y_lr, other.y_lr)
        else:
            y_ul = min(self.y_ul, other.y_ul)
            y_lr = max(self.y_lr, other.y_lr)
        px_ul, py_ul = gdal.ApplyGeoTransform(self.inv_geotransform, x_ul, y_ul)
        px_lr, py_lr = gdal.ApplyGeoTransform(self.inv_geotransform, x_lr, y_lr)
        px_size, py_size = int(px_lr - px_ul), int(py_lr - py_ul)
        if px_size < 1 or py_size < 1:
            return None
        _, x_res, x_skew, _, y_skew, y_res = self.geotransform
        return GeoFrame(crs=self.crs, geotransform=(x_ul, x_res, x_skew, y_ul, y_skew, y_res),
                        shape=(py_size, px_size))

    def __getitem__(self, bounds):
        """
        Returns the corresponding array values bounded by `bounds` parameter.
        :param bounds: GeoFrame, Window
        :return: numeric, numpy.ndarray
        """
        if isinstance(bounds, GeoFrame):
            bounds = self.window(bounds)
            if bounds is None: return None
        elif (isinstance(bounds, Window) and not bounds.is_valid()) or not isinstance(bounds, Window):
            raise IndexError('`bounds` is not a `geoframe` or a valid offset `window`.')
        return self.__array__()[bounds.index()]

    def __setitem__(self, bounds, value):
        """
        Sets corresponding array values bounded by `bounds` parameter.
        :param bounds: GeoFrame, Window
        :param value: numeric, numpy.ndarray
        """
        if isinstance(bounds, GeoFrame):
            bounds = self.window(bounds)
            if bounds is None:
                raise IndexError('informed `bounds` does not intersect with `geoframe`.')
        elif (isinstance(bounds, Window) and not bounds.is_valid()) or not isinstance(bounds, Window):
            raise IndexError('`bounds` is not a `geoframe` or a valid offset `window`.')
        self.__array__()[bounds.index()] = value

    def __array__(self):
        if self._buffer is None:
            n = self._height * self._width
            buffer = numpy.arange(n)
            buffer.shape = self.shape
            self._buffer = buffer
        return self._buffer

    def _verify_crs(self, other):
        if not self.crs.IsSame(other.crs):
            raise ValueError('`geoframe`s instances are not in the same projection.')

    def is_valid(self):
        return self.x_res > 0.0 and self.y_res > 0.0 and (self.crs.IsProjected() or self.crs.IsGeographic())

    def index(self):
        return self.window().index()

    def is_buffered(self):
        return self._buffer is not None

    def discard_buffer(self):
        self._buffer = None

    def window(self, other=None):
        """
        Returns a valid offset Window (px, py, px_size, py_size) from intersection of `self` and `bounds`.
        If no intersection, returns None.
        :param other: GeoFrame
        :return: Window
        """
        if other is None or self is other:
            py_size, px_size = self.shape
            return Window(0, 0, py_size, px_size)
        intersection = self * other
        if not intersection:
            return None
        px_ul, py_ul = gdal.ApplyGeoTransform(self.inv_geotransform, intersection.x_ul, intersection.y_ul)
        px_lr, py_lr = gdal.ApplyGeoTransform(self.inv_geotransform, intersection.x_lr, intersection.y_lr)
        px_size, py_size = int(px_lr - px_ul), int(py_lr - py_ul)
        if px_size < 0 or py_size < 0:
            return None
        return Window(int(px_ul), int(py_ul), px_size, py_size)

    def frame(self, bounds=None):
        """
        Returns a new GeoFrame with empty buffer bounding the given offset `window` or `geoframe`.
        :param bounds: GeoFrame, Window
        :return: GeoFrame
        """
        window = bounds
        if bounds is None or isinstance(bounds, GeoFrame):
            window = self.window(bounds)
        if not window.is_valid(): return None
        px, py, px_size, py_size = window()
        x_ul, y_ul = gdal.ApplyGeoTransform(self.geotransform, px, py)
        _, x_res, x_skew, _, y_skew, y_res = self.geotransform
        return GeoFrame(crs=self.crs, geotransform=(x_ul, x_res, x_skew, y_ul, y_skew, y_res),
                        shape=(py_size, px_size))

    def clip(self, bounds=None, copy=False):
        """
        Returns a corresponding GeoFrame intersection between `self` and `bounds` parameter.
        Resulting GeoFrame buffer is filled with corresponding clipped buffer.
        Parameter `copy` indicates if a new buffer is to be created with copied values.
        If no intersection, returns None.
        :param bounds: GeoFrame, Window
        :param copy: bool
        :return: GeoFrame
        """
        result = self.frame(bounds)
        if not result: return None
        if copy:
            result[result] = self[result].copy()
        else:
            result[result] = self[result]
        return result

    def stack_from(self, other):
        if self not in other:
            raise ValueError('informed `geoframe` argument does not cover all extent.')
        if self.is_buffered() and other.is_buffered():
            self._buffer = numpy.dstack((self._buffer, other[self]))
        elif other.is_buffered():
            self[self] = other[self].copy()
        else:
            raise ValueError('informed `geoframe` argument has not a valid buffer array.')

    def read_raster(self, raster, band_list=None, dtype='f8', resample_alg=GRIORA_NearestNeighbour, nodata=None):
        raster = open_raster(raster)
        if nodata is None:
            nodata = raster.GetRasterBand(1).GetNoDataValue()
        raster_frame = empty_geoframe_from_raster(raster)
        intersection = self * raster_frame
        if intersection is None: return None
        window = raster_frame.window(intersection)
        height, width = intersection.shape
        byte_str = raster.ReadRaster(xoff=window.px, yoff=window.py, xsize=window.px_size, ysize=window.py_size,
                                     buf_xsize=width, buf_ysize=height,
                                     buf_type=_gdal_dtype(dtype), band_list=band_list,
                                     resample_alg=resample_alg)

        data = numpy.fromstring(byte_str, dtype=dtype)
        data.shape = (-1, height, width)
        data = numpy.dstack(data)
        height, width = self.shape
        bands = data.shape[-1]
        if bands == 1:
            if self.is_buffered():
                if self._buffer.ndim == 2:
                    data.shape = data.shape[:2]
                self[intersection] = numpy.where(data == nodata, self[intersection], data)
            else:
                data.shape = data.shape[:2]
                self._buffer = numpy.ones((height, width)) * numpy.dtype(dtype).type(nodata)
                self[intersection] = data
        else:
            if self.is_buffered():
                if self._buffer.ndim == 2:
                    self._buffer.shape += (1,)
                self[intersection] = numpy.where(data == nodata, self[intersection], data)
            else:
                self._buffer = numpy.ones((height, width, bands)) * numpy.dtype(dtype).type(nodata)
                self[intersection] = data

    def write_raster(self, raster, dtype='u2', nodata=None):
        if not self.is_buffered():
            raise ValueError('`geoframe` has not a valid buffer array.')
        raster = open_raster(raster, GA_Update)
        bands = self.shape and self._bands
        if bands > raster.RasterCount:
            raise ValueError('`raster` has not sufficient bands.')
        raster_frame = empty_geoframe_from_raster(raster)
        intersection = self * raster_frame
        if intersection is None: return False
        from_window = self.window(intersection)
        to_window = raster_frame.window(intersection)

        if self._buffer.ndim == 3:
            buffer = self[intersection].astype(dtype, copy=False)
            height_size, width_size, bands_size = buffer.strides
        else:
            buffer = self[intersection].astype(dtype, copy=False)
            height_size, width_size = buffer.strides
            bands_size = 0
        raster.WriteRaster(xoff=to_window.px, yoff=to_window.py, xsize=to_window.px_size, ysize=to_window.py_size,
                           buf_string=buffer.astype(dtype, copy=False).tostring(),
                           buf_xsize=from_window.px_size, buf_ysize=from_window.py_size,
                           buf_type=_gdal_dtype(buffer.dtype),
                           buf_pixel_space=width_size, buf_line_space=height_size, buf_band_space=bands_size)
        # how to capture gdal errors?

        if nodata is not None:
            for i in range(1, bands + 1):
                raster.GetRasterBand(i).SetNodataValue(numpy.dtype(dtype).type(nodata))
        for i in range(1, raster.RasterCount + 1):
            raster.GetRasterBand(i).ComputeStatistics(False)
        raster.FlushCache()
        return True

    def create_raster(self, file_name, dtype='u2', nodata=None, gdal_driver='GTiff'):
        width, height = self.shape
        bands = self._bands
        result = create_raster(file_name, width, height, bands, dtype=dtype, gdal_driver=gdal_driver)
        result.SetGeoTransform(self.geotransform)
        if nodata is not None:
            for i in range(1, bands + 1):
                result.GetRasterBand(i).SetNodataValue(numpy.dtype(dtype).type(nodata))
        return result

    @property
    def shape(self):
        if self.is_buffered():
            if self._buffer.ndim == 3:
                self._height, self._width, self._bands = self._buffer.shape
            else:
                self._height, self._width = self._buffer.shape
                self._bands = 1
        height = self._height
        width = self._width
        return height, width

    @property
    def geotransform(self):
        return self.x_ul, self.x_res, self.x_skew, self.y_ul, self.y_skew, self.y_res

    @property
    def inv_geotransform(self):
        result = gdal.InvGeoTransform(self.geotransform)
        if result is None:
            raise TypeError('coordinates cannot be inverted to pixel space.')
        return result

    @property
    def x_lr(self):
        height, width = self.shape
        x_lr, _ = gdal.ApplyGeoTransform(self.geotransform, width, height)
        return x_lr

    @property
    def y_lr(self):
        height, width = self.shape
        _, y_lr = gdal.ApplyGeoTransform(self.geotransform, width, height)
        return y_lr


# import mygdal as geo
# import numpy
# a = numpy.arange(60)
# a.shape = (-1, 6)
# b = geo.GeoFrame(shape=a.shape)
# c=b.frame(geo.Window(2,2,3,3))
# b[c]
# b[c] = 1
# r=geo.open_raster('/dados/d3/rolf/MODIS_MOD13Q1/MOD13Q1.A2016209.h12v10.005.2016226031147_250m_16_days_EVI.tif')
# geo.copy_raster(r, '/dados/d2/rolf/teste.tif', overwrite=True)
# r=geo.open_raster('/dados/d2/rolf/teste.tif', geo.GA_Update)
# e=geo.empty_geoframe_from_raster(r)
# e.read_raster(r)
# e[geo.Window(1500,1500,1800,1800)] = -3000
# e.write_raster(r, dtype='i2')
# r=geo.copy_raster('/dados/d2/rolf/lsat_evi.tif', '/dados/d2/rolf/teste.tif', overwrite=True)
# e=geo.empty_geoframe_from_raster(r)
# e.read_raster(r)
# e.write_raster(r, dtype='i2')

