import numpy as np
from PIL import Image

#data shape is 115204 1 + 3 + 160*120*3 + 57600
# data shape should be 1 152 040
sampleLen = 115204
N = 100 # number of samples
file = "D:\\Projects\\originalDataSets\\originalDataTest.npy"
fileP = "D:\\Projects\\Nao\\database\\preprocessedDataTest.npy"
fileN = "D:\\Projects\\Nao\\database\\normalizedDataTest.npy"
fileM = "D:\\Projects\\Nao\\database\\maskedDataTest.npy"
rawData = np.load(file)
rawData = rawData.reshape(N, sampleLen)
ts = rawData[:, 0].astype(float)
joints = rawData[:, 1:4].astype(float)
topCam = rawData[:, 4:57604].astype(np.uint8)
botCam = rawData[:, 57604:].astype(np.uint8)

def convert(ar):
    rgb = ar.reshape(160, 120, 3)
    rgb = Image.fromarray(rgb)
    rgb = rgb.convert('L')
    img = rgb.resize((64, 64), Image.ANTIALIAS)
    return img

def mask(img):
    img = img.reshape(160, 120, 3)
    mask = (img[:,:,0]>100) & (img[:,:,1]>10) & (img[:,:,2]>30)
    mask = Image.fromarray(mask)
    mask = mask.resize((64, 64), Image.ANTIALIAS)
    return mask
   
def prepareDataToSave(im):
    im = np.asarray(im)
    return im.reshape(-1)

def preprocessData():
    data = []
    for i in range(N):
        data = np.append(data, ts[i])
        data = np.append(data, joints[i])
        data = np.append(data, prepareDataToSave(mask(topCam[i])))
        data = np.append(data, prepareDataToSave(mask(botCam[i])))
    np.save(fileM, data)


preprocessData()
