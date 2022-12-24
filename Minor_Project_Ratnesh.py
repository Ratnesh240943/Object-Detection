from tkinter import *
from PIL import Image, ImageTk
import cv2
import time
import os
import numpy as np

#Load YOLO Algorithm
net=cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

#To load all objects that have to be detected
classes=[]
with open("coco.names","r") as f:
    read=f.readlines()
for i in range(len(read)):
    classes.append(read[i].strip("\n"))

#Defining layer names
layer_names=net.getLayerNames()
output_layers=[]
for i in net.getUnconnectedOutLayers():
    output_layers.append(layer_names[i-1])

class Welcome(Tk):
    def __init__(self):
        super().__init__()
        #self.resizable(0,0)
        self.title("Photo Filter")
        self.configure(background="SkyBlue1")
        self.vid = MyVideoCapture()
        self.photos = Label(self)
        self.canvas	= Canvas(self.photos, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()
        self.photos.pack()
        self.geometry(f"{int(self.vid.width)}x{int(self.vid.height+50)}+0+0")
        self.resizable(0,0)
        self.picture = Frame(self,bg="SkyBlue1").pack(fill="both",expand=True)
        Button(self.picture, text="Capture",font=("Helvetica",15),width=30,relief=SOLID,bg="SkyBlue1",activebackground="SkyBlue1",activeforeground="blue",command=self.capture).pack()
        self.delay = 1
        self.update()

    def	capture(self):
        cv2.imwrite(r"Picture-"+time.strftime("%d-%m-%Y-%H-%M-%S")+'.jpg', self.frame)

    def	update(self):
        ret,frame, frame1=self.vid.get_frame()

        height,width,channels=frame.shape

        
        #Extracting features to detect objects
        blob=cv2.dnn.blobFromImage(frame,0.00392,(416,416),(0,0,0),True,crop=False)
                                                                #Inverting blue with red
                                                                #bgr->rgb


        #We need to pass the img_blob to the algorithm
        net.setInput(blob)
        outs=net.forward(output_layers)
        #print(outs)

        #Displaying informations on the screen
        class_ids=[]
        confidences=[]
        boxes=[]
        for output in outs:
            for detection in output:
                #Detecting confidence in 3 steps
                scores=detection[5:]                #1
                class_id=np.argmax(scores)          #2
                confidence =scores[class_id]        #3

                if confidence >0.5: #Means if the object is detected
                    center_x=int(detection[0]*width)
                    center_y=int(detection[1]*height)
                    w=int(detection[2]*width)
                    h=int(detection[3]*height)

                    #Drawing a rectangle
                    x=int(center_x-w/2) # top left value
                    y=int(center_y-h/2) # top left value

                    boxes.append([x,y,w,h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        #Removing Double Boxes
        indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.3,0.4)

        Person_count = 0
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]]  # name of the objects
                if(label == "person"):
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    Person_count += 1
                #else:
                    #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Person Count : " + str(Person_count), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        #frame = cv2.resize(frame, (self.vid.width,self.vid.height))


        if	ret:
            self.frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.photo=ImageTk.PhotoImage(image=Image.fromarray(self.frame))
            self.canvas.create_image(0,0,image=self.photo,anchor='nw')
        self.after(self.delay,self.update)

        self.frame = frame

class MyVideoCapture:
    def	__init__(self):
        self.vid=cv2.VideoCapture(0)
        if	not	self.vid.isOpened():
            print("Unable	to open	video source")
        
        self.width	= self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height	= self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frame1 = None


    def	get_frame(self):
        if	self.vid.isOpened():
            ret, frame = self.vid.read()
            if self.frame1 is None:
                self.frame1= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if ret:
                return (ret, frame,  self.frame1)
            else:
                return (ret, None)
        else:
            return	(False, None)

    def	__del__(self):
        if	self.vid.isOpened():
            self.vid.release()   

if __name__=="__main__":
    root = Welcome()
    root.mainloop()