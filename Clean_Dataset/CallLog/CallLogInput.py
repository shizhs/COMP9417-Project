import sys
sys.path.append('../../')

from Clean_Dataset.Utils.csv_util import csv_util

sms = csv_util('../../StudentLife_Dataset/Inputs/call_log/', True)
sms.readAll()
