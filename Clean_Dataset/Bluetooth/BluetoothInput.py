from pyspark import SparkConf
from pyspark import SparkContext as sc
import os

conf = SparkConf().setAppName('Shortest Path')
sc = sc(conf=conf)

inPath = './input/blue/'

files = os.listdir(inPath)
outputFile = []
for file in files:
    textFile = sc.textFile(inPath + file).filter(lambda s: not ('time' in s or '' == s))
    uid = file.replace('bt_', '').split('.')[0]
    step1 = textFile.map(lambda line: (line.split(",")[0], 1))
    step2 = step1.reduceByKey(lambda a, b: a + b)
    key = step2.count()
    value = sum(step2.values().collect())
    outputFile.append((uid, str(value // key)))

s = ''

outputFile.sort()

total = 0
for t in outputFile:
    total += int(t[1])
mean = str(total // len(outputFile))
print('Mean is ' + mean)

lack = []

index = 0
front = outputFile[index]
for t in outputFile[1:]:
    fid = int(front[0].replace('u', ''))
    tid = int(t[0].replace('u', ''))
    while fid + 1 != tid:
        fid = fid + 1
        if fid < 10:
            lack.append(('u0' + str(fid), mean))
        else:
            lack.append(('u' + str(fid), mean))
        print('Add a lack uid: u' + str(fid))
    index += 1
    front = outputFile[index]

outputFile.extend(lack)
outputFile.sort()

outputFile.insert(0, ('uid', 'frequency'))

for t in outputFile:
    s += t[0] + ',' + t[1] + '\n'
with open('./output/bluetooth.csv', 'w') as file:
    file.write(s)
