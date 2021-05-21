from objects import Detection 
from main import *
import numpy as np 

bbox1 = np.array([[0,0,50,50],[150,150,200,200]])
classes1 = np.array([0,1])
scores1 = np.array([0.5,0.5])

detection1 = Detection(bbox1,classes1,scores1)

bbox2 = np.array([[10,10,60,60],[250,250,400,400]])
classes2 = np.array([0,1])
scores2 = np.array([0.5,0.5])

detection2 = Detection(bbox2,classes2,scores2)

img = np.zeros((500,500,3))

detection1.visualize(img,'Current')
detection2.visualize(img,'next')
ans =  track_sequence(detection1.detection(), detection2.detection())
print(ans)
