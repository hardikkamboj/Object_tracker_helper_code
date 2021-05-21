import numpy as np
import cv2 

COLOR_CODING = {0: (255,0,0), 1: (0,255,0), 2: (0,0,255)}
class Detection:
    '''
        - `bboxes` {np.ndarray} (n_detections, 4) shaped array of bboxes with 4 eles [xmin, xmax, ymin, ymax]
            eg, for 5 detections
                [[xmin,  ymin, ,xmax, ymax],
                 [xmin,  ymin, ,xmax, ymax],
                 [xmin,  ymin, ,xmax, ymax],
                 [xmin,  ymin, ,xmax, ymax],
                 [xmin,  ymin, ,xmax, ymax]]
        - `classes` {np.ndarray} length n_detections which has one of 3 unique class ids for every bbox
            eg, [0,2,0,1,1]
        - `scores` {list} length n_detections between [0,1]
            eg, [0.4,0.2,0.0,0.1,0.9]
    '''
    def __init__(self,bbox,classes,scores):
        self.bbox = bbox
        self.classes = classes
        self.scores = scores

        self.detection_dict = {'bboxes':self.bbox,
                         'classes':self.classes,
                         'scores':self.scores}

    def bbox(self):
        return self.bbox

    def classes(self):
        return self.classes 

    def scores(self):
        return self.scores

    def detection(self):
        return self.detection_dict

    def visualize(self,img,msg):
        image = np.copy(img)
        for i in range(self.bbox.shape[0]):
            x1,y1,x2,y2 = self.bbox[i]
            color  = COLOR_CODING[self.classes[i]] 
            image = cv2.rectangle(image,(x1,y1),(x2,y2),color,3)
            cv2.imshow(msg,image)
        cv2.waitKey(0)

