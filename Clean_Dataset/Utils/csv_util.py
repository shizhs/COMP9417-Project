import pandas as pd
import glob

class csv_util:
    '''some utils for reading and writing csv'''

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
        self._log(self.csv_files)

        for f in self.csv_files:
            self.readOne(f)

    def readOne(self, path):
        '''read one csv file and do something with it'''
        csv = pd.read_csv(path)
        self._log(csv)

    def writeToCsv(self):
        '''write output to a csv file'''

    def _log(self, *args, **kwargs):
        """print if debug is true"""
        if (self.debug):
            print(*args, **kwargs)
