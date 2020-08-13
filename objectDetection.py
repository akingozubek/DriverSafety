import cv2
import numpy as np

#net=cv2.dnn.readNet("Models/yolov3.weights","Models/yolov3.cfg")
net=cv2.dnn.readNet("Models/yolov3-tiny.weights","Models/yolov3-tiny.cfg")

classes=[]

with open("Models/coco.names","r") as f:
    classes=f.read().splitlines()
#with open("smoke.names","r") as f:
#    classes=f.read().splitlines()



cam=cv2.VideoCapture(0)

while True:
    ret,img=cam.read()
    height,width,_=img.shape

    blob=cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0),swapRB=True,crop=False)
    net.setInput(blob)
    out_layers_name=net.getUnconnectedOutLayersNames()
    layer_outs=net.forward(out_layers_name)

    boxes=[]
    confidences=[]
    class_ids=[]

    for out in layer_outs:
        for detection in out:
            score=detection[5:]
            class_id=np.argmax(score)
            confidence=score[class_id]
            if confidence>0.5:
                center_x=int(detection[0]*width)
                center_y=int(detection[0]*height)
                
                w=int(detection[2]*width)
                h=int(detection[2]*height)
                
                x=int(center_x-w/2)
                y=int(center_y-h/2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
            
    idx=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
    font=cv2.FONT_HERSHEY_PLAIN
    colors=np.random.uniform(0,255,size=(len(boxes),3))
    try:
        for i in idx.flatten():
            x,y,w,h=boxes[i]
            label=str(classes[class_ids[i]])
            confidence=str(round(confidences[i],2))
            color=colors[i]
            cv2.rectangle(img,(x,y),(x+w,y+h),color,1)
            cv2.putText(img,label+" "+confidence,(x,y+20),font,2,(255,255,255),2)
    except:
        pass    
    cv2.imshow("Camera",img)

    k=cv2.waitKey(10) & 0xff
    if k==27:
        break
cam.release()
cv2.destroyAllWindows()