import os
import sys
import tarfile


def landsat_file_fields(file):
    try:
        # LXSPPPRRRYYYYDDDGSIVV
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

        # PPP = WRS path
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
        result = result and prod_band_f in ('sr_ndvi', 'sr_evi', 'cfmask', 'cfmask_conf')
        result = result and ext_f in 'tif'
        if result:
            return lsat_f, sensor_f, sat_f, wrs_path_f, wrs_row_f, year_f, doy_f, gsi_f, ver_f, prod_band_f, ext_f
        else:
            raise ValueError('Invalid landsat file name format.')
    except ValueError:
        raise ValueError('Invalid landsat file name format.')


class Repository:
    def __init__(self, path):
        self.path = path
        self.__paths__ = []
        self.__files__ = []
        self.__fields__ = []
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith('.tif'):
                    try:
                        lsat_f, sensor_f, sat_f, wrs_path_f, wrs_row_f, year_f, doy_f, gsi_f, \
                            ver_f, prod_band_f, ext_f = landsat_file_fields(file)
                        self.__paths__.append(root)
                        self.__files__.append(file)
                        self.__fields__.append({'sensor': sensor_f, 'satellite': sat_f, 'path': wrs_path_f,
                                               'row': wrs_row_f, 'year': year_f, 'doy': doy_f, 'GSI': gsi_f,
                                               'ver': ver_f, 'prod_band': prod_band_f})
                    except ValueError:
                        pass
        self.entities = list(set([file[:21] for file in self.__files__]))

    def