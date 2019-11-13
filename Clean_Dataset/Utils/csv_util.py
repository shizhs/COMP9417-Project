import pandas as pd
import numpy as np
import glob

class csv_util:
    '''
    some utils for reading and writing csv
    - call readAll first
    - call process and pass a function in for it
    '''

    def __init__(self, path, debug = False):
        '''
        path is a folder containing csv files,
        dedug enables printing
        '''
        self.path = path
        self.debug = debug

        # save a list of csv files
        self.csv_files = []
        # the output we need to write to csv
        self.output = []

    def readAll(self):
        '''read and save all csv file names under that folder'''
        self.csv_files = glob.glob(self.path + '/*.csv')
        # self._log(self.csv_files)

    def process(self, func):
        '''process all data'''
        for path in self.csv_files:
            # get uid (-4 to remove csv extension)
            uid = path.split('_')[-1][:-4]
            # self._log(uid)
            csv = pd.read_csv(path)
            percentage = func(csv)
            self.output.append([uid, percentage])
        self._log(self.output)


    def writeToCsv(self, titles, csv_name):
        '''write output to a csv file'''
        data = pd.DataFrame(np.array(self.output), columns=titles)
        data.to_csv(csv_name, index=False)

    def _log(self, *args, **kwargs):
        """print if debug is true"""
        if (self.debug):
            print(*args, **kwargs)
