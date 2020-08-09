import numpy as np
from PIL import Image

#############             PART ZU DATA EINLESEN               #################
#data shape is (8196, N of samples) :
#1(time steps) + 3(joints) + 64*64(topCam) + 4096(one image length, botCam)

sampleLen = 8196
N = 2000 # number of samples
file = "D:\\Projects\\Nao\\database\\originalDataTrain.npy"
fileP = "D:\\Projects\\Nao\\database\\preprocessedDataTrain.npy"
fileN = "D:\\Projects\\Nao\\database\\normalizedDataTrain.npy"
fileM = "D:\\Projects\\Nao\\database\\normalizedDataTest.npy"
rawData = np.load(fileM)
rawData = rawData.reshape(N, sampleLen)
ts = rawData[:, 0].astype(float)
joints = rawData[:, 1:4].astype(float)
topCam = rawData[:, 4:4100].astype(float) # to visualize image should be changed to np.uint8
botCam = rawData[:, 4100:].astype(float)

###################################################################################

mean = np.mean(topCam)
std = np.std(topCam)

def normalize():
    for row in rawData:
        for i in range(4,sampleLen):
            row[i] = (row[i]- mean) / std
    return rawData

print(botCam)