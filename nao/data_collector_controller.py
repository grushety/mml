"""data_collector controller."""
import random
import time
import csv
import numpy as np
from controller import Robot, Camera, PositionSensor

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# get and enable Camera
cameraTop = Camera("CameraTop")
cameraTop.enable(timestep)

print("Width: ", cameraTop.getWidth())
print("Height: ", cameraTop.getHeight())
#cameraTop.saveImage("cameraImg.png", 50)

cameraBottom = Camera("CameraBottom")
cameraBottom.enable(timestep)

#Move head in right position
#HeadYaw = robot.getMotor("HeadYaw")
#HeadPitch = robot.getMotor("HeadPitch")

#HeadYaw.setPosition(0.5)
#HeadYaw.setVelocity(1.0)
#HeadPitch.setPosition(0.2)
#HeadPitch.setVelocity(1.0)

# define all motors we going to use


LShoulderPitch = robot.getMotor("LShoulderPitch")
LShoulderRoll = robot.getMotor("LShoulderRoll")
LElbowRoll = robot.getMotor("LElbowRoll")

# get and enable position sensors
LShoulderPitchS = robot.getPositionSensor("LShoulderPitchS")
LShoulderRollS = robot.getPositionSensor("LShoulderRollS")
LElbowRollS = robot.getPositionSensor("LElbowRollS")

LShoulderPitchS.enable(timestep)
LShoulderRollS.enable(timestep)
LElbowRollS.enable(timestep)

# get and enable sonars
SonarR = robot.getDistanceSensor("Sonar/Right")
SonarL = robot.getDistanceSensor("Sonar/Left")

SonarR.enable(timestep)
SonarL.enable(timestep)


#test movement

LShoulderPitch.setVelocity(1.0)
LShoulderRoll.setVelocity(1.0)
LElbowRoll.setVelocity(1.0)

def getPosition():
    return [round(LShoulderPitchS.getValue(),2),round(LShoulderRollS.getValue(),2), round(LElbowRollS.getValue(),2)]
    
def generateRandomConfig():
    return [round(random.uniform(-0.5, 0.5), 2), round(random.uniform(-0.31, 0.7), 2), round(random.uniform(-1.54, 0.0), 2)]
    
def startMovement(config):
    LShoulderPitch.setPosition(config[0])
    LShoulderRoll.setPosition(config[1])
    LElbowRoll.setPosition(config[2])
    
    
def getSample():
    ts = time.time()
    position=np.array(getPosition(), dtype=np.float)
    dataPlate = np.append(ts, position)
    imageTop = np.array(cameraTop.getImageArray(), dtype=np.int)
    dataPlate = np.append(dataPlate, imageTop.reshape(-1))
    imageBottom = np.array(cameraBottom.getImageArray(), dtype=np.int)
    dataPlate = np.append(dataPlate, imageBottom.reshape(-1))
    # data shape is 115204 1 + 3 + 160*120*3 + 57600
    return dataPlate
    
def saveData(samples):
    #Change path 
    fileName = "C:\\Users\\Patrick\\Dropbox\\Yulia\\study\\Multimodal Machine Learning\\originalData.tsv"
    with open(fileName, 'w', encoding='utf8', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        tsv_writer.writerow(["Timestep", "JointConfigs", "RGB_Image_Top", "RGB_Image_Bottom"])
        for sample in samples:
            tsv_writer.writerow(sample)
config = generateRandomConfig()
startMovement(config)

def saveCSV(samples):
    file = "/home/yulia/Dropbox/Yulia/study/Multimodal Machine Learning/originalDataTest.npy"
    np.save(file, samples)

timer = time.time()+ 0.1
oldPos = [0.0, 0.0, 0.0]
sampleN = 0
sampleMax = 1000
samples = np.array([])
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1 and sampleN != sampleMax:
    #print("Sonar Left:", SonarL.getValue())
    #print("Sonar Right:", SonarR.getValue())
    currentPos = getPosition()  
    if time.time() > timer: 
        if (currentPos == oldPos):
            config = generateRandomConfig()
            startMovement(config)
            timer = time.time() + 2
    oldPos = currentPos
    samples = np.append(samples, getSample(), axis=0)
    sampleN +=1          
    print(sampleN)   
#saveCSV(samples)
