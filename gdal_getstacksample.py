# -*- coding: utf-8 -*-
from mygdal import Samples, Timeline
from scipy import interpolate
import numpy
import matplotlib.pyplot as plt

if __name__ == '__main__':
    s = Samples('../data/samples_new.csv')
    s.fetch_data()
    class_indexes = s.get_data_key_indexes('class')
    # print(class_indexes)
    samples_index = class_indexes["Forest"][:15]
    timeseries = s.get_samples_timeseries(samples_index, True)
    print(timeseries)
    for i in range(len(samples_index)):
        for j in range(len(s.bands)):
            values = timeseries[i][j][0]
            days = timeseries[i][j][1]
            class_value = s.data[s.tags[Samples.TAG_CLASS_FIELD]][samples_index[i]]
            f = interpolate.interp1d(days, values)
            days_new = numpy.linspace(1, 360, 22, True)
            plt.plot(days, values, 'o', days_new, f(days_new), '-')
    # sources = s.timeline.get_data_key_indexes('source')
    # print(s.timeline.data[1][sources['LT5']])

    plt.legend(['ndvi', 'linear', 'cubic', 'evi', 'linear', 'cubic'], loc='best')
    plt.title(class_value)
    plt.show()
    s.close()
