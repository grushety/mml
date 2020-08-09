import csv

timesteps=[]
joint_config=[]
topCam=[]
botCam=[]
csv.field_size_limit(100000000)

file = "C:\\Users\\Patrick\\Dropbox\\Yulia\\study\\Multimodal Machine Learning\\originalData.csv"
with open(file, 'r', encoding='utf-8',newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=" ")
    next(reader)
    for row in reader:
        timesteps = row[0]
        joint_config= row[1]
        topCam = row[2]
print (topCam)