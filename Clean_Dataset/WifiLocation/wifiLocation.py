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
    ret = str(mean[1]).replace(', ','|').replace('[', '').replace(']', '')
    for m in mean[2:]:
        ret = ret + ',' + str(m).replace(', ','|').replace('[', '').replace(']', '')
    return start + ',' + ret

conf = SparkConf().setAppName('Shortest Path')
sc = sc(conf=conf)

inPath = './input/wifi_location/'
ouPath = './result/'
files = os.listdir(inPath)
outputFile = []

for file in files:
    textFile = sc.textFile(inPath + file).filter(lambda s: not ('time' in s or '' == s))
    uid = file.replace('wifi_location_', '').split('.')[0]
    step1 = textFile.map(lambda line: ((getWeek(line.split(",")[0]), line.split(",")[1].split("[")[0]), [1, {getDay(line.split(",")[0])}]))
    step2 = step1.reduceByKey(lambda a, b: [a[0] + b[0], a[1] | b[1]])
    # key = step2.count()
    # value = sum(step2.values().collect())
    dic = step2.sortByKey().collectAsMap() # ((3,0), 100)
    outList = [[0, 0], [0, 0], [0, 0], [0, 0],
               [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
               [0, 0], [0, 0]]
    for v in dic.keys():
            index = 0 if v[1].__eq__('in') else 1
            outList[int(v[0])][index] = dic[v][0] // len(dic[v][1])
    s = uid
    for i in outList[1:]:
        s = s + ',' + str(i).replace(', ','|').replace('[', '').replace(']', '')
    outputFile.append((uid, s))
    # if uid.__eq__('u05'):
    #     step2.saveAsTextFile(ouPath)
    #     with open('./output/u15.csv', 'w') as file1:
    #         file1.write(s)
s = ''

outputFile.sort()

# total = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
#                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
#                [0, 0, 0, 0], [0, 0, 0, 0]]
# mean = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
#                [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
#                [0, 0, 0, 0], [0, 0, 0, 0]]
# for i in range(1, 11):
#     for t in outputFile:
#         for p in range(0, 4):
#             total[i][p] += int(t[1].split(',')[i].split('|')[p].replace('[','').replace(']',''))
#     for p in range(0, 4):
#         mean[i][p] = total[i][p] // len(outputFile)
# print(mean)
# print(total)
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
with open('./output/wifiLocationDay.csv', 'w') as file:
    file.write(s)

