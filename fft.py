import os
import zipfile
import gzip
import shutil

folder = 'data'

for item in os.listdir(folder):
    if item.endswith(".gz"):
        print('/' + folder + '/' + item)
        with gzip.open(folder + '/' + item, 'rb') as f_in:
            with open(folder + '/' + item[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

