from objects import Detection
import numpy as np 


MAX_PXL_DIST_BETWEEN_SERIAL_ASSETS = {
    0: 50,
    1: 50,
    2: 50
}


def serial_tracker_v1(cur_asset_bbox, nxt_asset_bbox, dist_thresh, check_side='right'):
    """
    :@param cur_asset_box: {ndarray}
        [x1,y1,x2,y2] representing the parameters of the box
            eg, [0,0,50,50] represents a box at upper right corner of side 50

    :@param nxt_asset_box: {ndarray}
        [x1,y1,x2,y2] representing the parameters of the box
    
    :@param dist_thresh: {int}
        threshold value

    :@param check_side: {string}
        'right' or 'left' 
        right - checks only if the box at nxt_asset_bbox is at the right side of the box at cur_asset_box
    """
    points1 = cur_asset_bbox
    points2 = nxt_asset_bbox

    
    #middle point of the boxes
    middle_cur = np.array((points1[0] + points1[2])/2, (points1[1] + points1[3]) / 2)
    middle_nxt = np.array((points2[0] + points2[2])/2, (points2[1] + points2[3]) / 2)

    if ( (check_side == 'right' and not points2[1] > points1[1]) or (check_side == 'left' and points2[1] < points1[1]) ):
        return False

    dist = np.sqrt((points2[0] - points1[0])**2 + (points2[1] - points1[1])**2 )
    print(dist)

    return True if dist < dist_thresh else False
    

def serial_tracker_v2(cur_asset_bbox, possible_bboxes_for_same_class, dist_thresh, check_side='right'):
    pass


def track_sequence(cur_detections, next_detections):
    """
    :@param cur_detections: {dict}
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
    
    :@param next_detections: {dict}
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
    """
    tracked_assets = dict()

    for unique_cur_asset_id, cur_asset_label in enumerate(cur_detections['classes']):
        
        cur_asset_bbox = cur_detections['bboxes'][unique_cur_asset_id]
        

        for unique_nxt_asset_id, nxt_asset_label in enumerate(next_detections['classes']):
            nxt_asset_bbox = next_detections['bboxes'][unique_nxt_asset_id]

            if not (cur_asset_label == nxt_asset_label):
                continue

            # core logic: comparing for same class bboxes
            is_same_asset = serial_tracker_v1(cur_asset_bbox, nxt_asset_bbox, dist_thresh=MAX_PXL_DIST_BETWEEN_SERIAL_ASSETS[cur_asset_label])
            
            if is_same_asset:
                tracked_assets[unique_cur_asset_id] = unique_nxt_asset_id

    return tracked_assets