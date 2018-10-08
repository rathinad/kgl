import numpy as np

def printvalues(aa, bb):
    
    print('Entering printvalues')
    sumtotal=0.0
    sumtotal2=0.0
    scores          = np.zeros((0,))
    
    #aaa=numpy.array([numpy.array(xi) for xi in aa])
    
    # bbb=numpy.array([numpy.array(xi) for xi in bb])
    
    for d in aa:
        print('Printing first detection')
        scores = np.append(scores, d[4])
        sumtotal = map_iou(np.expand_dims(d, axis=0), bb, scores)
        print ('Printing map for image: ')
        print(sumtotal)
        sumtotal2+=sumtotal
    print('printing total sum')
    print(sumtotal2)
    #print('length of detections')
    #print(len(aa))
    print('Exiting printvalues')
    
    
    
    
# helper function to calculate IoU
def iou(box1, box2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22, sc = box2
    #x11, y11, w1, h1 = box1
    #x21, y21, w2, h2, sc = box2
    #assert w1 * h1 > 0
    #assert w2 * h2 > 0
    #x12, y12 = x11 + w1, y11 + h1
    #x22, y22 = x21 + w2, y21 + h2

    #area1, area2 = w1 * h1, w2 * h2
    area1, area2 = (x12-x11) * (y12-y11), (x22-x21) * (y22-y21)
    xi1, yi1, xi2, yi2 = max([x11, x21]), max([y11, y21]), min([x12, x22]), min([y12, y22])
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    else:
        intersect = (xi2-xi1) * (yi2-yi1)
        union = area1 + area2 - intersect
        return intersect / union
    
# simple test
#box1 = [100, 100, 200, 200]
#box2 = [100, 100, 300, 200]
#print(iou(box1, box2))


def map_iou(boxes_true, boxes_pred, scores, thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]):
    """
    Mean average precision at differnet intersection over union (IoU) threshold
    
    input:
        boxes_true: Mx4 numpy array of ground true bounding boxes of one image. 
                    bbox format: (x1, y1, w, h)
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image. 
                    bbox format: (x1, y1, w, h)
        scores:     length N numpy array of scores associated with predicted bboxes
        thresholds: IoU shresholds to evaluate mean average precision on
    output: 
        map: mean average precision of the image
    """
    
    # According to the introduction, images with no ground truth bboxes will not be 
    # included in the map score unless there is a false positive detection (?)
        
    # return None if both are empty, don't count the image in final evaluation (?)
    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        return None
    
    assert boxes_true.shape[1] == 4 or boxes_pred.shape[1] == 4, "boxes should be 2D arrays with shape[1]=4"
    if len(boxes_pred):
        assert len(scores) == len(boxes_pred), "boxes_pred and scores should be same length"
        # sort boxes_pred by scores in decreasing order
        boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]
    
    map_total = 0
    
    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
        tp, fn = 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                #print('Printing j, bt and bp')
                #print(j)
                #print(bt)
                #print(bp)
                miou = iou(bt, bp)
                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1 # bt is matched for the first time, count as TP
                    matched_bt.add(j)
            if not matched:
                fn += 1 # bt has no match, count as FN
                
        fp = len(boxes_pred) - len(matched_bt) # FP is the bp that not matched to any bt
        m = tp / (tp + fn + fp)
        #print('Threshold : %s tp : %s fn : %s fp : %s map %s' % (t,tp,fn,fp,m))
        map_total += m
    
    k =  map_total / len(thresholds)
    #print ('About to return')
    #print(k)
    return k

# simple test
#boxes_true = np.array([[100, 100, 200, 200]])
#boxes_pred = np.array([[100, 100, 300, 200]])
#scores = [0.9]

#map_iou(boxes_true, boxes_pred, scores)