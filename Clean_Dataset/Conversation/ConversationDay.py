from pyspark import SparkConf
from pyspark import SparkContext as sc
import os


def getWeek(stamp):
    startStamp = 1364256000.0
    day = (float(stamp) - startStamp) / 60 / 60 // 24 + 1
    week = day // 7 + 1
    return int(week)


def getDay(stamp):
    startStamp = 1364256000.0
    day = (float(stamp) - startStamp) / 60 / 60 // 24 + 1
    return int(day)


def meanToString(mean, start):
    ret = str(mean[1])
    for m in mean[2:]:
        ret = ret + ',' + str(m)
    return start + ',' + ret


conf = SparkConf().setAppName('Shortest Path')
sc = sc(conf=conf)

inPath = './input/conversation/'
ouPath = './result/'
files = os.listdir(inPath)
outputFile = []

for file in files:
    textFile = sc.textFile(inPath + file).filter(lambda s: not ('time' in s or '' == s))
    uid = file.replace('conversation_', '').split('.')[0]
    step1 = textFile.map(lambda line: (getWeek(line.split(",")[0]), [int(line.split(",")[1]) // 60 -
                                                                     int(line.split(",")[0]) // 60, {getWeek(line.split(",")[0])}]))
    step2 = step1.reduceByKey(lambda a, b: [a[0] + b[0], a[1] | b[1]])
    # key = step2.count()
    # value = sum(step2.values().collect())
    dic = step2.sortByKey().collectAsMap()
    outList = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for v in dic.keys():
        outList[v] = dic[v][0] // len(dic[v][1])
    s = uid
    for i in outList[1:]:
        s = s + ',' + str(i).replace(', ', '|').replace('[', '').replace(']', '')
    outputFile.append((uid, s))
    if uid.__eq__('u05'):
        step2.saveAsTextFile(ouPath)
        with open('./output/u15.csv', 'w') as file1:
            file1.write(s)
s = ''

outputFile.sort()

# total = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# mean = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# for i in range(1, 11):
#     for t in outputFile:
#         total[i] += int(t[1].split(',')[i])
#     mean[i] = str(total[i] // len(outputFile))
# print(mean)
#
# lack = []
#
# index = 0
# front = outputFile[index]
# for t in outputFile[1:]:
#     fid = int(front[0].replace('u', ''))
#     tid = int(t[0].replace('u', ''))
#     while fid + 1 != tid:
#         fid = fid + 1
#         if fid < 10:
#             lack.append(('u0' + str(fid), meanToString(mean, 'u0' + str(fid))))
#         else:
#             lack.append(('u' + str(fid), meanToString(mean, 'u' + str(fid))))
#         print('Add a lack uid: u' + str(fid))
#     index += 1
#     front = outputFile[index]
#
# outputFile.extend(lack)
# outputFile.sort()

outputFile.insert(0, ('key', 'uid,week1,week2,week3,week4,week5,week6,week7,week8,week9,week10'))

for t in outputFile:
    s += t[1] + '\n'
with open('./output/conversationDay.csv', 'w') as file:
    file.write(s)


# u00,539,428,555,933,897,910,627,584,316,1533
# u01,362,276,220,400,1002,374,298,478,339,2795
# u02,479,936,867,1244,1168,1413,2003,1538,2515,6621
# u03,207,363,285,165,652,288,350,341,155,0
# u04,483,779,787,1109,1056,863,1017,682,598,0
# u05,540,1040,969,1138,1678,2321,3211,3702,2433,42384
# u06,394,774,810,1093,1144,1124,1238,4067,6422,9226
# u07,847,1013,1390,1784,1486,1469,838,334,0,0
# u08,948,693,906,1630,1365,1729,2484,4015,2174,2156
# u09,32,513,605,860,1279,1214,2334,1647,682,4676
# u10,837,1524,1835,2303,2083,1441,1900,1482,1752,923
# u11,394,774,810,1093,1144,1124,1238,4067,6422,9226
# u12,204,417,393,675,584,513,1035,916,1374,590
# u13,202,1397,1065,1948,3039,7169,4371,54404,64120,34463
# u14,62,835,1493,1741,1605,2344,1529,1235,1578,28179
# u15,388,401,720,804,613,912,545,536,22,472
