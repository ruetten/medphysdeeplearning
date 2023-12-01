import csv
from datetime import datetime as dt

def parseData(ptid):
    with open("LP_ADNIMERGE.csv", newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            row = row[0].split(",")
            if row[2] == ptid and (row[0] == "AD" or row[0] == "CN"):
                # There are 2 date formats in the spreadsheet. We must accept both
                p1 = "%m/%d/%Y"
                p2 = "%d-%m-%Y"
                epoch = dt(1970, 1, 1)
                
                try:
                    print((dt.strptime(row[4], p1) - epoch).total_seconds())
                except:
                    print((dt.strptime(row[4], p2) - epoch).total_seconds())

parseData(input("PTID from CSV: "))