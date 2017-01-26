# -*- coding: utf-8 -*-

import tarfile
import sys
import os

if len(sys.argv) < 3:
    print('Usage: %s <from_files>' % sys.argv[0])
    exit()

target_directory = './'
from_files = sys.argv[1:]
for i, file in enumerate(from_files):
    if file.endswith('.tar.gz'):
        print('%s. Extracting %s...' % (i, file), end=' ')
        tar = tarfile.open(file, 'r:gz')
        tar.extractall(path=target_directory)
        tar.close()
        print('Done!')
print('Finished!')
