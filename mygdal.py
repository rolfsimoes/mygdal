from osgeo import gdal
from osgeo import osr
import gdalnumeric
import numpy
import os

# Pixel conversion
PIXEL_FLOOR = 0
PIXEL_CEIL = 1

# Location conversion
LOC_FLOOR = 0
LOC_CENTER = 1

# GeoTransform constants indexes
GT_X_UL = 0
GT_X_RES = 1
GT_X_SKEW = 2
GT_Y_UL = 3
GT_Y_SKEW = 4
GT_Y_RES = 5

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

# gdal resample algorithms' constants
GRA_NearestNeighbour = gdal.GRA_NearestNeighbour
GRA_Bilinear = gdal.GRA_Bilinear
GRA_Cubic = gdal.GRA_Cubic
GRA_CubicSpline = gdal.GRA_CubicSpline
GRA_Lanczos = gdal.GRA_Lanczos
GRA_Average = gdal.GRA_Average
GRA_Mode = gdal.GRA_Mode

# gdal dataset open mode
GF_Read = gdal.GF_Read
GF_Write = gdal.GF_Write


def get_proj(anyproj, return_proj4=False):
    srs = osr.SpatialReference()
    if isinstance(anyproj, int):
        srs.ImportFromEPSG(anyproj)
    else:
        srs.SetFromUserInput(anyproj)
    if not srs.IsGeographic() and not srs.IsProjected():
        raise ValueError('informed `anyproj` argument could not be resolved.')
    if return_proj4:
        return srs.ExportToProj4()
    return srs.ExportToWkt()


def get_srs(proj):
    result = osr.SpatialReference()
    result.ImportFromProj4(proj)
    if not result.IsGeographic() and not result.IsProjected():
        raise ValueError('informed `any_proj` argument could not be resolved.')
    return result


def create(width=100, height=50, bands=3, proj=None, nodata=None, gdal_dtype=GDT_UInt16):
    result = raster(dataset=None)
    result.__init_new__(width=width, height=height, bands=bands, proj=proj, nodata=nodata, gdal_dtype=gdal_dtype)
    return result


def open_file(filename):
    result = raster(dataset=None)
    result.__init_file__(filename=filename)
    return result


class raster_band:
    def __init__(self, raster_source, band_num, nodata=None, empty=False):
        self.raster = raster_source
        self.band = raster_source.dataset.GetRasterBand(band_num)
        if nodata is None:
            nodata = self.band.GetNoDataValue()
        self.set_nodata(nodata)
        self.__empty__ = empty
        self.__data_array__ = None

    def __load_band__(self):
        if self.__data_array__ is None:
            if self.__empty__:
                self.__data_array__ = numpy.ones((self.raster.height, self.raster.width),
                                                 gdalnumeric.GDALTypeCodeToNumericTypeCode(self.data_type)) * \
                                                 self.nodata
            else:
                data = numpy.empty((self.raster.height, self.raster.width),
                                   gdalnumeric.GDALTypeCodeToNumericTypeCode(self.data_type))
                self.__data_array__ = self.band.ReadAsArray(xoff=0, yoff=0,
                                                            win_xsize=self.raster.width, win_ysize=self.raster.height,
                                                            buf_obj=data)

    def __save_band__(self):
        if self.raster and self.__data_array__ is not None:
            self.band.WriteArray(self.__data_array__)

    def __getitem__(self, item):
        return self.array[item]

    def __setitem__(self, key, value):
        self.__load_band__()
        self.__data_array__[key] = value

    @property
    def array(self):
        self.__load_band__()
        return self.__data_array__

    @property
    def nodata(self):
        return self.band.GetNoDataValue()

    @property
    def shape(self):
        return self.array.shape

    @property
    def data_type(self):
        return self.band.DataType

    def set_nodata(self, value):
        if value is None:
            value = 0
        self.band.SetNoDataValue(value)

    def fill(self, value):
        self[:] = value


class raster:
    def __init__(self, dataset=None):
        self.dataset = dataset
        self.bands = []
        self.nodata_bands = numpy.array([])
        if dataset:
            self.bands = [raster_band(raster_source=self, band_num=i) for i in range(1, self.dataset.RasterCount + 1)]
            self.nodata_bands = numpy.array([band.nodata for band in self.bands])

    def __init_file__(self, filename):
        self.dataset = gdal.Open(filename)
        if not self.dataset:
            raise IOError('file `{0}` is invalid or missing.'.format(filename))
        self.bands = [raster_band(raster_source=self, band_num=i) for i in range(1, self.dataset.RasterCount + 1)]
        self.nodata_bands = numpy.array([band.nodata for band in self.bands])

    def __init_new__(self, width, height, bands, proj=None, nodata=None, gdal_dtype=GDT_UInt16):
        self.dataset = gdal.GetDriverByName('MEM').Create(utf8_path='', xsize=width, ysize=height,
                                                          bands=bands, eType=gdal_dtype)
        if not self.dataset:
            raise IOError('an error occurred while creating gdal `MEM` dataset.')
        self.bands = [raster_band(raster_source=self, band_num=i, nodata=nodata, empty=True)
                      for i in range(1, self.dataset.RasterCount + 1)]
        self.nodata_bands = numpy.array([band.nodata for band in self.bands])
        if proj is None:
            proj = '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs'
        self.set_proj(proj)
        self.set_bounds(x_ul=-180.0, y_lr=-90.0, x_lr=180.0, y_ul=90.0)

    def __save__(self):
        for band in self.bands:
            band.__save_band__()
        self.dataset.FlushCache()

    def __repr__(self):
        result = 'raster<{0}x{1}x{2}>'.format(self.height, self.width, len(self.bands))
        return result

    @property
    def driver(self):
        return self.dataset.GetDriver().ShortName

    @property
    def data_type(self):
        if len(self.bands):
            return self.bands[0].data_type
        return None

    @property
    def width(self):
        return self.dataset.RasterXSize

    @property
    def height(self):
        return self.dataset.RasterYSize

    @property
    def proj(self):
        return get_proj(anyproj=self.dataset.GetProjectionRef(), return_proj4=True)

    @property
    def bounds(self):
        x_ul, x_res, x_skew, y_ul, y_skew, y_res = self.dataset.GetGeoTransform()
        return x_ul, y_ul + self.height * y_res + self.width * y_skew, \
            x_ul + self.width * x_res + self.height * x_skew, y_ul

    @property
    def geotransform(self):
        return self.dataset.GetGeoTransform()

    def set_proj(self, anyproj):
        self.dataset.SetProjection(get_proj(anyproj=anyproj))

    def set_bounds(self, x_ul, y_lr, x_lr, y_ul, x_skew=0, y_skew=0):
        new_geo_transform = (x_ul, (x_lr - x_ul - self.height * x_skew) / self.width,
                             x_skew, y_ul, y_skew, (y_lr - y_ul - self.width * y_skew) / self.height)
        self.dataset.SetGeoTransform(new_geo_transform)

    def set_geotransform(self, x_ul, x_res, x_skew, y_ul, y_skew, y_res):
        new_geo_transform = (x_ul, x_res, x_skew, y_ul, y_skew, y_res)
        self.dataset.SetGeoTransform(new_geo_transform)

    def set_metadata(self, args):
        self.dataset.SetMetadata(args)

    def to_pixel(self, x, y, pixel_round=PIXEL_FLOOR):
        success, px_ul, px_res, px_skew, py_ul, py_skew, py_res = gdal.InvGeoTransform(self.geotransform)
        if not success:
            raise TypeError('coordinates cannot be inverted to pixel space.')
        px = px_ul + x * px_res + y * px_skew
        px = int(px) + pixel_round if (px - int(px)) > 0.0125 else 0
        py = py_ul + y * py_res + x * py_skew
        py = int(py) + pixel_round if (py - int(py)) > 0.0125 else 0
        return py, px

    def to_location(self, py, px, loc_round=LOC_FLOOR):
        x_ul, x_res, x_skew, y_ul, y_skew, y_res = self.geotransform
        x = x_ul + px * x_res + py * x_skew + loc_round * x_res / 2
        y = y_ul + py * y_res + px * y_skew + loc_round * y_res / 2
        return x, y

    def copy_to(self, target, to_bands=None, resample_alg=GRA_Cubic):
        if to_bands is None:
            to_bands = range(len(self.bands))
        elif isinstance(to_bands, int):
            if to_bands >= len(target.bands):
                raise IndexError('band index is out of range.')
            to_bands = range(to_bands, to_bands + len(self.bands))
        if len(to_bands) > len(self.bands):
            raise ValueError('there is not enough bands to copy.')

        x_ul, y_lr, x_lr, y_ul = target.bounds
        _, x_res, x_skew, _, y_skew, y_res = target.geotransform
        result = self.warp(x_ul=x_ul, y_lr=y_lr, x_lr=x_lr, y_ul=y_ul,
                           x_res=x_res, y_res=y_res, to_anyproj=target.proj,
                           resampling=resample_alg)
        for i, to_band in enumerate(to_bands):
            if to_band >= len(target.bands):
                raise IndexError('band index is out of range.')
            target.bands[to_band][:] = numpy.where(target.bands[to_band].array == target.bands[to_band].nodata,
                                                   result.bands[i].array, target.bands[to_band].array)
        return result

    def warp(self, x_ul, y_lr, x_lr, y_ul, x_res, y_res, to_anyproj=None, resampling=GRA_NearestNeighbour,
             out_type=gdal.GDT_Unknown):
        if to_anyproj is None:
            to_anyproj = self.proj
        else:
            to_anyproj = get_proj(anyproj=to_anyproj)
        str_nodata_bands = str(self.nodata_bands)[1:-1]
        options = gdal.WarpOptions(format='MEM', outputType=out_type,
                                   srcSRS=self.proj, dstSRS=to_anyproj,
                                   outputBounds=[x_ul, y_lr, x_lr, y_ul],
                                   xRes=x_res, yRes=y_res,
                                   targetAlignedPixels=False,
                                   resampleAlg=resampling,
                                   srcNodata=str_nodata_bands, dstNodata=str_nodata_bands,
                                   multithread=True, errorThreshold=0.125)
        dataset = gdal.Warp(destNameOrDestDS='', srcDSOrSrcDSTab=self.dataset, options=options)
        result = raster(dataset=dataset)
        return result

    def save_as(self, file, overwrite=False, gdal_format='GTiff'):
        if not self.dataset:
            raise IOError('invalid or undefined dataset.')
        if not overwrite and os.path.isfile(file):
            raise IOError('file already exists.')
        driver = gdal.GetDriverByName(gdal_format)
        metadata = driver.GetMetadata()
        if not (gdal.DCAP_CREATECOPY in metadata and metadata[gdal.DCAP_CREATECOPY] == 'YES'):
            raise IOError('format `{}` does not support copy.'.format(gdal_format))
        self.__save__()
        driver.CreateCopy(file, self.dataset, False)

    def read_pixel(self, py, px, factor_value=1.0, default_value=numpy.nan, min_value=None, max_value=None, dtype='f8'):
        result = self.dataset.ReadAsArray(xoff=px, yoff=py, xsize=1, ysize=1)
        result = numpy.reshape(result, len(self.bands))
        good_data = result != self.nodata_bands
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
