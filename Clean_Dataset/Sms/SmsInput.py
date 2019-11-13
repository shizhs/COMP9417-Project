import sys
sys.path.append('../../')

from Clean_Dataset.Utils.csv_util import csv_util

def percentage(data):
    '''data is a DataFrame'''
    return 0.1

sms = csv_util('../../StudentLife_Dataset/Inputs/sms/', True)
sms.readAll()
sms.process(percentage)
sms.writeToCsv(['uid', 'percentage'], 'sms.csv')
