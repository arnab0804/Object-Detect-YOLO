import cv2 as cv
import numpy as np

fileName="coco.names"
classNames=[]
with open(fileName,"rt") as file:
    classNames=file.read().rstrip("\n").split("\n")

threshold=0.5
weightPath="yolov3.weights"
configPath="yolov3.cfg"
net=cv.dnn.readNet(weightPath,configPath)

layer_names=net.getLayerNames()
output_layers=[layer_names[i-1] for i in net.getUnconnectedOutLayers()]

def rescaleFrame(frame,scale_x=1.2,scale_y=1.2):
    breadth=int(frame.shape[1]*scale_x)
    length=int(frame.shape[0]*scale_y)
    dimensions=(breadth,length)

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

vid_capture=cv.VideoCapture(0)
while True:
    isTrue,frame=vid_capture.read()
    changed_frame=rescaleFrame(frame)
    height,width,channels=changed_frame.shape

    blob=cv.dnn.blobFromImage(changed_frame,0.00392,(416,416),(0,0,0),True,crop=False)
    net.setInput(blob)
    outputs=net.forward(output_layers)

    boxes=[]
    confidences=[]
    classIds=[]
    for out in outputs:
        for detecton in out:
            conf_scores=detecton[5:]
            classId=np.argmax(conf_scores)
            confidence=conf_scores[classId]
            if confidence>threshold:
                center_x=int(detecton[0]*width)
                center_y=int(detecton[1]*height)
                w=int(detecton[2]*width)
                h=int(detecton[3]*height)

                x=int(center_x-w/2)
                y=int(center_y-h/2)
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                classIds.append(classId)
    
    indexes=cv.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
    for i in indexes:
        x,y,w,h=boxes[i]
        object_label=str(classNames[classIds[i]])
        object_conf=str(confidences[i])
        cv.rectangle(changed_frame,(x,y),(x+w,y+h),(0,255,0),thickness=2)
        cv.putText(changed_frame,object_label,(x+10,y+20),fontFace=cv.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(0,255,0),thickness=1)
        cv.putText(changed_frame,str(round(float(object_conf)*100,2)),(x+200,y+20),fontFace=cv.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(0,255,0),thickness=1)

    cv.imshow("Camera Capture",changed_frame)                
    
    if cv.waitKey(20) & 0xFF==ord('d'):
        break