import sys
sys.path.append('../../')

from Clean_Dataset.Utils.csv_util import csv_util

sms = csv_util('../../StudentLife_Dataset/Inputs/sms/', True)
sms.readAll()
