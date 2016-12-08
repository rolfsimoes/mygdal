# -*- coding: utf-8 -*-
from mygdal import Samples

if __name__ == '__main__':
    s = Samples('samples_new.csv')
    s.fetch_data()
    class_indexes = s.get_data_key_indexes('class')
    s.get_samples_timeseries()
    s.close()
