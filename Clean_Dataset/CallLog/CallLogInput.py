# handle import here
import sys
sys.path.append('../../')

'''
SmsInput
'''

from Clean_Dataset.Utils.csv_util import csv_util
import pandas as pd
import numpy as np

def percentage(data):
    '''data is a DataFrame'''

    array = np.array(data)
    call_count = 0
    for row in array:
        for value in row[6:]:
            if not str(value) == "nan":
                call_count += 1
                break

    length = len(array)
    if length == 0:
        return 0.0
    return call_count / length

sms = csv_util('../../StudentLife_Dataset/Inputs/call_log/')
sms.readAll()
sms.process(percentage)
sms.writeToCsv(['uid', 'percentage'], 'call_log.csv')
