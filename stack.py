# -*- coding: utf-8 -*-

import geoarray as geo
import os
import mysat

lsat_path = '/dados/d3/rolf/LANDSAT_EVI_NDVI/'
modis_path = '/dados/d3/rolf/MODIS_MOD13Q1/'


def my_new_lsat_stack(bands, dtype, nodata, **metadata):
    return geo.empty(width=320, height=268, bands=bands,
                     geotransform=(608775.00, 30.0, 0.0, -1324875.00, 0.0, -30.0),
                     proj_origin='+proj=utm +zone=21 +ellps=WGS84 +datum=WGS84 +units=m +no_defs',
                     nodata=nodata, dtype=dtype, **metadata)


def my_new_lsat_sensor_stack(lsat_rep_band, save_in_path):

    def sensor_to_int(x):
        return 5 if x == 'LT5' else 7 if x == 'LE7' else 8 if x == 'LC8' else 0

    # sensor stack
    sensor_band = lsat_rep_band.apply(source_int=(sensor_to_int, 'source'))
    t = my_new_lsat_stack(bands=lsat_rep_band.rows, dtype='u1', nodata=0, SEMANTIC='5 LT5, 7 LE7, 8 LC8')
    for i in range(lsat_rep_band.rows):
        t[t, i] = sensor_band['source_int'][i]
    geo.write(t, file_name=os.path.join(save_in_path, 'lsat_{0}.tif'.format('source')), overwrite=True)


def my_new_lsat_days_stack(lsat_rep_band, save_in_path):
    # days after 1970-01-01 stack
    t = my_new_lsat_stack(bands=lsat_rep_band.rows, dtype='u2', nodata=0, SEMANTIC='DAYS SINCE 1970-01-01')
    for i in range(lsat_rep_band.rows):
        t[t, i] = lsat_rep_band['datestamp'][i]
    geo.write(t, file_name=os.path.join(save_in_path, 'lsat_{0}.tif'.format('days')), overwrite=True)


def make_stack_lsat_only(root, sensor=True, days=True, **bands):
    repository = mysat.landsat(root)
    repository = repository.order_by('date')

    one_band = repository.where(repository.index_of('product', 'sr_ndvi'))

    # sensor stack
    if sensor:
        my_new_lsat_sensor_stack(lsat_rep_band=one_band, save_in_path='/dados/d2/rolf')
        print('{0}: {1} lsat_bands'.format('sensor', one_band.rows))

    # days after 1970-01-01 stack
    if days:
        my_new_lsat_days_stack(lsat_rep_band=one_band, save_in_path='/dados/d2/rolf')
        print('{0}: {1} lsat_bands'.format('days', one_band.rows))

    # bands stack
    for band_name, band in bands.items():
        print('-----------------------------------')
        one_band = repository.where(repository.index_of('product', band))
        t = my_new_lsat_stack(bands=one_band.rows, dtype='i2', nodata=-9999, SEMANTIC=band_name)
        for i in range(one_band.rows):
            s = geo.read(one_band['file'][i], t.box, resample_alg=geo.GRIORA_NearestNeighbour)
            t[s, i] = s[t, 0]
            print('{0} band {1:02d}: {2}'.format(band_name, i, one_band['file'][i]))
        if t is not None:
            geo.write(t, file_name='/dados/d2/rolf/lsat_{0}.tif'.format(band_name), overwrite=True)


def brick_modis_only(root, sensor=True, days=True, **bands):
    repository = mysat.mod13q1(root)
    repository = repository.order_by('date')

    one_band = repository.where(repository.index_of('product', '250m_16_days_NDVI'))

    # sensor stack
    if sensor:
        my_new_lsat_sensor_stack(lsat_rep_band=one_band, save_in_path='/dados/d2/rolf')
        print('{0}: {1} mod13q1_bands'.format('sensor', one_band.rows))

    # days after 1970-01-01 stack
    if days:
        my_new_lsat_days_stack(lsat_rep_band=one_band, save_in_path='/dados/d2/rolf')
        print('{0}: {1} mod13q1_bands'.format('days', one_band.rows))

    # bands stack
    for band_name, band in bands.items():
        print('-----------------------------------')
        one_band = repository.where(repository.index_of('product', band))
        t = my_new_lsat_stack(bands=one_band.rows, dtype='i2', nodata=-9999, SEMANTIC=band_name)
        for i in range(one_band.rows):
            s = geo.read(one_band['file'][i], t.box, resample_alg=geo.GRIORA_NearestNeighbour)
            t[s, i] = s[t, 0]
            print('{0} band {1:02d}: {2}'.format(band_name, i, one_band['file'][i]))
        if t is not None:
            geo.write(t, file_name='/dados/d2/rolf/lsat_{0}.tif'.format(band_name), overwrite=True)
########
    for band_name, bands in modis_bands.items():
        print('-----------------------------------')
        paths = []
        dates = []
        for band in bands:
            files = [file for file in os.listdir(band[1]) if file.endswith(band[0] + '.tif')]
            paths += [os.path.join(band[1], file) for file in files]
            dates += [band[2](file) for file in files]
        paths_dates = sorted(zip(paths, dates), key=lambda px: px[1])

        t = mygdal.create(px_size=320, py_size=268, bands=len(paths_dates),
                          proj='+proj=utm +zone=21 +ellps=WGS84 +datum=WGS84 +units=m +no_defs',
                          nodata=-3000,
                          gdal_dtype=mygdal.GDT_Int16,
                          geotransform=(608775.00, 30.0, 0.0, -1324875.00, 0.0, -30.0))

        for i in range(len(paths_dates)):
            s = mygdal.open_file(filename=paths_dates[i][0])
            print('{0} band {1:02d}: {2}'.format(band_name, i, paths_dates[i][0]))
            s.copy_to(target=t, to_bands=i, resample_alg=mygdal.GRA_NearestNeighbour)
        t.save_as('/dados/d2/rolf/modis_{0}.tif'.format(band_name), overwrite=True)


make_stack_lsat_only(root=lsat_path, sensor=True, days=True, ndvi='sr_ndvi', evi='sr_evi')

#
# def brick_lsat_modis():
#     all_bands = {'ndvi': [('NDVI', modis_path, modis_extr_date, modis_extr_sensor),
#                           ('sr_ndvi', lsat_path, lsat_extr_date, lsat_extr_sensor)],
#                  'evi': [('EVI', modis_path, modis_extr_date, modis_extr_sensor),
#                          ('sr_evi', lsat_path, lsat_extr_date, lsat_extr_sensor)]}
#     paths = []
#     dates = []
#     sensor = []
#     for band in all_bands['ndvi']:
#         files = [file for file in os.listdir(band[1]) if file.endswith(band[0] + '.tif')]
#         paths += [os.path.join(band[1], file) for file in files]
#         dates += [band[2](file) for file in files]
#         sensor += [file[0:3] for file in files]
#
#     paths_dates = sorted(zip(paths, dates, sensor), key=lambda px: px[1])
#
#     # sensor stack
#     t = mygdal.create(px_size=320, py_size=268, bands=len(paths_dates),
#                       proj='+proj=utm +zone=21 +ellps=WGS84 +datum=WGS84 +units=m +no_defs',
#                       gdal_dtype=mygdal.GDT_Byte,
#                       geotransform=(608775.00, 30.0, 0.0, -1324875.00, 0.0, -30.0))
#     for i in range(len(paths_dates)):
#         if paths_dates[i][2] == 'LT5':
#             t.bands[i].fill(5)
#         elif paths_dates[i][2] == 'LE7':
#             t.bands[i].fill(7)
#         elif paths_dates[i][2] == 'LC8':
#             t.bands[i].fill(8)
#         elif paths_dates[i][2] == 'MOD':
#             t.bands[i].fill(113)
#     t.save_as('/dados/d2/rolf/lsat_modis_{0}.tif'.format('sensor'), overwrite=True)
#     print('{0}: {1} lsat_bands'.format('sensor', len(paths_dates)))
#
#     # days after 1970-01-01 stack
#     t = mygdal.create(px_size=320, py_size=268, bands=len(paths_dates),
#                       proj='+proj=utm +zone=21 +ellps=WGS84 +datum=WGS84 +units=m +no_defs',
#                       gdal_dtype=mygdal.GDT_UInt16,
#                       geotransform=(608775.00, 30.0, 0.0, -1324875.00, 0.0, -30.0))
#     for i in range(len(paths_dates)):
#         if paths_dates[i][2] != 'MOD':
#             year = '{0}-01-01'.format(paths_dates[i][1][0:4])
#             days = int(paths_dates[i][1][4:7])
#             days_after_1970 = mynumpy.datetime64_to_timestamp(numpy.datetime64(year, 'D') +
#                                                               numpy.timedelta64(days, 'D')) / 86400
#             t.bands[i].fill(days_after_1970)
#         else:
#             year = '{0}-01-01'.format(paths_dates[i][1][0:4])
#             s = mygdal.open_file(paths_dates[i][0][:len(modis_path) + 55] + 'composite_day_of_the_year.tif')
#             s.copy_to(target=t, to_bands=i, resample_alg=mygdal.GRA_NearestNeighbour)
#             days = t.bands[i].array.astype('timedelta64[D]')
#             days_after_1970 = mynumpy.datetime64_to_timestamp(numpy.datetime64(year, 'D') + days - 1) / 86400
#             t.bands[i][:] = days_after_1970
#         t.set_metadata({'Semantic': 'days since 1970-01-01'})
#     t.save_as('/dados/d2/rolf/lsat_modis_{0}.tif'.format('days'), overwrite=True)
#     print('{0}: {1} lsat_bands'.format('days', len(paths_dates)))
#
#     for band_name, bands in all_bands.items():
#         print('-----------------------------------')
#         paths = []
#         dates = []
#         for band in bands:
#             files = [file for file in os.listdir(band[1]) if file.endswith(band[0] + '.tif')]
#             paths += [os.path.join(band[1], file) for file in files]
#             dates += [band[2](file) for file in files]
#         paths_dates = sorted(zip(paths, dates), key=lambda px: px[1])
#
#         t = mygdal.create(px_size=320, py_size=268, bands=len(paths_dates),
#                           proj='+proj=utm +zone=21 +ellps=WGS84 +datum=WGS84 +units=m +no_defs',
#                           nodata=-9999,
#                           gdal_dtype=mygdal.GDT_Int16,
#                           geotransform=(608775.00, 30.0, 0.0, -1324875.00, 0.0, -30.0))
#
#         for i in range(len(paths_dates)):
#             s = mygdal.open_file(filename=paths_dates[i][0])
#             print('{0} band {1:02d}: {2}'.format(band_name, i, paths_dates[i][0]))
#             s.copy_to(t, to_bands=i)
#         t.save_as('/dados/d2/rolf/lsat_modis_{0}.tif'.format(band_name), overwrite=True)
#
#

