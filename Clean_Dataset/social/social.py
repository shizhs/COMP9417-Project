import os 
import json
import csv

with open("social.csv","a",newline="") as f:
    csv_write = csv.writer(f, dialect='excel')
    title = ["uid",">5person","social times"]
    csv_write.writerow(title)
    for file in os.listdir("./response/Social"):
        dirt = "./response/Social/" + file
        with open(dirt,"r") as f:
            load_json = json.load(f)
        nb = len(load_json)
        if nb != 0:
            use_data_len = 0 #count all data used
            count=0
            uid = file[-8:-5]
            for i in load_json:
                if "null" in i :
                    continue
                use_data_len += 1
                if int(i["number"]) >= 2: #social >5person
                    count += 1
            if use_data_len != 0:
                percent_5more = count/use_data_len
                print(uid,percent_5more)
                stu = [uid,percent_5more,use_data_len]
                csv_write.writerow(stu)
