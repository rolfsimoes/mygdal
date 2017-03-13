import mydas
import numpy

samples = mydas.from_recarray(numpy.recfromcsv('../data/samples_new2.csv',
                                              dtype=[('id', 'i4'),
                                                     ('longitude', 'f8'),
                                                     ('latitude', 'f8'),
                                                     ('from', 'datetime64[D]'),
                                                     ('to', 'datetime64[D]'),
                                                     ('label', 'U14')]))
samples['geoloc'] = numpy.array([samples['longitude'], samples['latitude']]).T
samples.order_by(key='latitude', reverse=True)

labels = {}
for i, value in enumerate(numpy.unique(samples['label'])):
    labels[value] = i
    print(str(i), value)
