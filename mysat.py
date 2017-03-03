import os
import mydas
import numpy
import mynumpy


class landsat(mydas.mydas):
    def __init__(self, root):
        super().__init__()
        self.root = root
        file_paths = []
        entities = []
        source = []
        year = []
        doy = []
        date = []
        prod_band = []
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.endswith('.tif'):
                    try:
                        entity_f, lsat_f, sensor_f, sat_f, wrs_path_f, wrs_row_f, year_f, doy_f, gsi_f, \
                            ver_f, prod_band_f, ext_f = self.landsat_file_fields(file)
                        file_paths.append(os.path.join(root, file))
                        entities.append(entity_f)
                        source.append(lsat_f + sensor_f + sat_f)
                        year.append(numpy.datetime64('{}-01-01'.format(year_f), 'Y'))
                        doy.append(int(doy_f))
                        date.append(numpy.datetime64('{}-01-01'.format(year_f), 'D') +
                                    numpy.timedelta64(int(doy_f), 'D'))
                        prod_band.append(prod_band_f)
                    except ValueError:
                        continue
        self['file'] = file_paths
        self['entity'] = entities
        self['source'] = source
        self['year'] = year
        self['doy'] = doy
        self['date'] = numpy.array(date, dtype='M8[D]')
        self['datestamp'] = numpy.array(mynumpy.datetime64_to_timestamp(self['date']) // 86400, dtype='i2')
        self['product'] = prod_band

    @staticmethod
    def landsat_file_fields(file):
        try:
            # LXSPPPRRRYYYYDDDGSIVV
            entity_f = file[:21]

            # L = Landsat
            lsat_f = file[0]

            # X = Sensor
            sensor_f = file[1]
            # C = OLI & TIRS, O = OLI only, T = TIRS only
            # E = ETM+, T = TM, M = MSS

            # S = Satellite
            sat_f = file[2]
            # 8 = Landsat-8, 7 = Landsat-7, 5 = Landsat-5, 4 = Landsat-4
            # 3 = Landsat-3, 2 = Landsat-2, 1 = Landsat-1

            # PPP = WRS lsat_path
            wrs_path_f = int(file[3:6])

            # RRR = WRS row
            wrs_row_f = int(file[6:9])

            # YYYY = Year
            year_f = int(file[9:13])

            # DDD = Julian day of year
            doy_f = int(file[13:16])

            # GSI = Ground station identifier
            gsi_f = file[16:19]

            # VV = Archive version number
            ver_f = int(file[19:21])

            # File Product/Content
            prod_band_f = file[22:-4]

            # File extension
            ext_f = file[-3:]

            result = lsat_f == 'L'
            if sat_f in '123':
                result = result and (1 <= int(wrs_path_f) <= 251) and (1 <= int(wrs_row_f) <= 119) and sensor_f == 'M'
                if sat_f == '1':
                    result = result and (1972 <= year_f <= 1978)
                elif sat_f == '2':
                    result = result and (1975 <= year_f <= 1983)
                else:
                    result = result and (1978 <= year_f <= 1983)
            elif sat_f in '4578':
                result = result and (1 <= int(wrs_path_f) <= 233) and (1 <= int(wrs_row_f) <= 248)
                if sat_f in '45':
                    result = result and sensor_f in 'MT'
                    if sat_f == '4':
                        result = result and (1982 <= year_f <= 2001)
                    else:
                        result = result and (1984 <= year_f <= 2013)
                elif sat_f == '7':
                    result = result and sensor_f in 'E' and (1999 <= year_f)
                elif sat_f == '8':
                    result = result and sensor_f in 'COT' and (2013 <= year_f)
                else:
                    result = False
            result = result and (1 <= doy_f <= 366)
            result = result and gsi_f in ('COA', 'ASA', 'HOA', 'CUB', 'GNC', 'PAC', 'BJC', 'KHC', 'SNC', 'CPE', 'LBG',
                                          'NSG', 'SGI', 'DKI', 'RPI', 'FUI', 'MTI', 'KUJ', 'HAJ', 'HIJ', 'MLK', 'BIK',
                                          'CHM', 'ULM', 'ISP', 'IKR', 'MGR', 'MOR', 'RSA', 'JSA', 'MPS', 'KIS', 'CLT',
                                          'BKT', 'SRT', 'UPR')
            result = result and (0 <= ver_f)
            result = result and prod_band_f in ('band1', 'band2', 'band3', 'band4',
                                                'band5', 'band6', 'band7',
                                                'cfmask', 'cfmask_conf',
                                                'sr_adjacent_cloud_qa', 'sr_atmos_opacity',
                                                'sr_band1', 'sr_band2', 'sr_band3', 'sr_band4',
                                                'sr_band5', 'sr_band6', 'sr_band7',
                                                'sr_cloud_qa', 'sr_cloud_shadow_qa',
                                                'sr_ddv_qa', 'sr_fill_qa', 'sr_land_water_qa',
                                                'sr_snow_qa', 'sr_ndvi', 'sr_evi', )
            result = result and ext_f in ('tif',)
            if result:
                return entity_f, lsat_f, sensor_f, sat_f, wrs_path_f, wrs_row_f, year_f, doy_f, gsi_f, ver_f, prod_band_f, ext_f
            else:
                raise ValueError('Invalid landsat file name format.')
        except ValueError:
            raise ValueError('Invalid landsat file name format.')


class mod13q1(mydas.mydas):
    def __init__(self, root):
        super().__init__()
        self.root = root
        file_paths = []
        entities = []
        source = []
        year = []
        doy = []
        date = []
        prod_band = []
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.endswith('.tif'):
                    try:
                        entity_f, data_name_f, year_f, doy_f, htn_f, vtn_f, coll_f, \
                            pdt_f, prod_band_f, ext_f = self.mod13q1_tif_file_fields(file)
                        file_paths.append(os.path.join(root, file))
                        entities.append(entity_f)
                        source.append(data_name_f)
                        year.append(numpy.datetime64('{}-01-01'.format(year_f), 'Y'))
                        doy.append(int(doy_f))
                        date.append(numpy.datetime64('{}-01-01'.format(year_f), 'D') +
                                    numpy.timedelta64(int(doy_f), 'D'))
                        prod_band.append(prod_band_f)
                    except ValueError:
                        continue
        self['file'] = file_paths
        self['entity'] = entities
        self['source'] = source
        self['year'] = year
        self['doy'] = doy
        self['date'] = numpy.array(date, dtype='M8[D]')
        self['datestamp'] = numpy.array(mynumpy.datetime64_to_timestamp(self['date']) // 86400, dtype='i2')
        self['product'] = prod_band

    @staticmethod
    def mod13q1_tif_file_fields(file):
        try:
            # MOD13Q1.AYYYYDDD.hHHvVV.CCC.YYYYDDDHHMMSS_~.tif
            entity_f = file[:41]

            # MOD13Q1 = Earth Science Data Type Name
            data_name_f = file[0:7]

            # YYYY = Year of acquisition
            year_f = int(file[9:13])

            # DDD = Julian day of year
            doy_f = int(file[13:16])

            # HH = Horizontal tile number (0-35)
            htn_f = int(file[18:20])

            # VV = Vertical tile number (0-17)
            vtn_f = int(file[21:23])

            # CCC = Collection number
            coll_f = int(file[24:27])

            # YYYYDDDHHMMSS = Production Date and Time
            pdt_f = int(file[28:41])

            # File Product/Content
            prod_band_f = file[42:file.index('.tif')]

            # File extension
            ext_f = file[-3:]

            result = data_name_f == 'MOD13Q1'
            result = result and (2000 <= year_f)
            result = result and (1 <= doy_f <= 366)
            result = result and (0 <= htn_f <= 35)
            result = result and (0 <= vtn_f <= 17)
            result = result and coll_f >= 5
            result = result and prod_band_f in ('250m_16_days_blue_reflectance',
                                                '250m_16_days_composite_day_of_the_year'
                                                '250m_16_days_EVI',
                                                '250m_16_days_MIR_reflectance',
                                                '250m_16_days_NDVI',
                                                '250m_16_days_NIR_reflectance',
                                                '250m_16_days_pixel_reliability',
                                                '250m_16_days_red_reflectance',
                                                '250m_16_days_relative_azimuth_angle',
                                                '250m_16_days_sun_zenith_angle',
                                                '250m_16_days_view_zenith_angle',
                                                '250m_16_days_VI_Quality')
            result = result and ext_f in ('tif',)
            if result:
                return entity_f, data_name_f, year_f, doy_f, htn_f, vtn_f, coll_f, pdt_f, prod_band_f, ext_f
            else:
                raise ValueError('invalid mod13q1 file name format.')
        except ValueError:
            raise ValueError('invalid mod13q1 file name format.')
