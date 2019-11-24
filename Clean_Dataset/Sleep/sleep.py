import pandas as pd

light = pd.read_csv('../Dark/darkDay.csv')
plock = pd.read_csv('../PhoneLock/phonelockDay.csv')
pcharge = pd.read_csv('../PhoneCharge/phonechargeDay.csv')
stationary = pd.read_csv('../Activity/activityStationary.csv')
slience = pd.read_csv('../Audio/audioSlience.csv')

slienceCo = 0.3484
stationaryCo = 0.5445
pchargeCo = 0.0469
plockCo = 0.0512
lightCo = 0.0415

print(light.shape)
print(plock.shape)
print(pcharge.shape)
print(stationary.shape)
print(slience.shape)

line, column = light.shape

s = 'uid,week1,week2,week3,week4,week5,week6,week7,week8,week9,week10\n'
print(light)
for i in range(line):
    if i >= 1:
        s = s + light.iat[i, 0]
    for j in range(column):
        if j >= 1:
            sleepTime = lightCo*light.iat[i, j] + plockCo*plock.iat[i, j] + pchargeCo*pcharge.iat[i, j] \
                        + stationaryCo*stationary.iat[i, j] + slienceCo*slience.iat[i, j]
            sleepTime = sleepTime / 60
            s = s + ',' + str(round(sleepTime, 2))
    s = s + '\n'

with open('./sleep.csv', 'w') as file:
    file.write(s)

