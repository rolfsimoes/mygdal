# -*- coding: utf-8 -*-
from mygdal import Samples, Timeline
from scipy import interpolate
import numpy
import matplotlib.pyplot as plt

if __name__ == '__main__':
    s = Samples('../__data__/samples_new.csv')
    s.fetch_data()
    label_dict_indexes = s.get_dict_key_indexes('label')
    # print(class_indexes)
    print(label_dict_indexes.cols())
    class_value = 'Soybean-maize'
    my_samples = label_dict_indexes[class_value]

    #
    # timeseries = s.get_samples_timeseries(my_samples, True)
    # for i in range(len(my_samples)):
    #     for j in range(len(s.attrs)):
    #         values = timeseries[i][j][0]
    #         days = timeseries[i][j][1]
    #         ordered = days.argsort()
    #         f = interpolate.interp1d(days[ordered], values[ordered])
    #         days_new = numpy.linspace(15, 330, 22, True)
    #         plt.plot(days[ordered], values[ordered], 'o', days_new, f(days_new), '-')
    # sources = s.timeline.get_data_key_indexes('source')
    # print(s.timeline.data[1][sources['LT5']])

    # plt.legend(['ndvi', 'linear', 'cubic', 'evi', 'linear', 'cubic'], loc='best')
    plt.title(class_value)
    plt.show()
    s.close()
