import numpy as np

#data shape is 115204 1 + 3 + 160*120*3 + 57600
# data shape should be 1 152 040
sampleLen = 115204
N = 100 # number of samples
file = "D:\\Projects\\Nao\\database\\normalizedDataTest.npy"
rawData = np.load(file)
rawData = rawData.reshape(N, sampleLen)
ts = rawData[:, 0].astype(float)
joints = rawData[:, 1:4].astype(float)
topCam = rawData[:, 4:57604].astype(int)
botCam = rawData[:, 57605:].astype(int)
print (botCam)