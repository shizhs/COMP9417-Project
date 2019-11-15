import os 
import json
import csv

with open("event.csv","a",newline="") as f:
    csv_write = csv.writer(f, dialect='excel')
    title = ["uid","event_times"]
    csv_write.writerow(title)
    for file in os.listdir("./response/Events"):
        dirt = "./response/Events/" + file
        with open(dirt,"r") as f:
            load_json = json.load(f)
    
        use_data_len = 0 #count all data used
        uid = file[-8:-5]
        for i in load_json:
            use_data_len += 1
        stu = [uid,use_data_len]
        csv_write.writerow(stu)
