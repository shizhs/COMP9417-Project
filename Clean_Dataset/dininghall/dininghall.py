import os 
import json
import csv

with open("dininghall.csv","a",newline="") as f:
    csv_write = csv.writer(f, dialect='excel')
    title = ["uid","bf_incampus_pct","lc_incampus_pct"]
    csv_write.writerow(title)
    for file in os.listdir("./response/Dining Halls"):
        dirt = "./response/Dining Halls/" + file
        with open(dirt,"r") as f:
            load_json = json.load(f)
        nb = len(load_json)
        if nb != 0:
            count_bf = 0
            count_lc = 0
            uid = file[-8:-5]
            for i in load_json:
                if i["breakfast"] == "7": #off campus
                    count_bf += 1
                if i["lunch"] == "7": #off campus
                    count_lc +=1
            #breakfast in campus
            percent_bf = 1 - count_bf/nb
            #lunch in campus
            percent_lc = 1- count_lc/nb
            stu = [uid,percent_bf,percent_lc]
            csv_write.writerow(stu)
