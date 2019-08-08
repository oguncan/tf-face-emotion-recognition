# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 14:55:00 2019

@author: Cigdem Karaman and Ogun Can KAYA
"""


from PyQt5.QtWidgets import QApplication, QFrame,QPushButton, QHBoxLayout, QGroupBox, QVBoxLayout,QLabel,QDialog,QMessageBox
import sys
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtCore import QRect,QTimer
from PyQt5 import QtCore
from PIL import Image
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
import os
import tensorflow as tf
from align_custom import AlignCustom
from face_feature import FaceFeature
from mtcnn_detect import MTCNNDetect
from tf_graph import FaceRecGraph
import time
import YOLO_OBJECT
import json
from PyQt5.uic import loadUi
# %%
TIMEOUT = 10 #10 seconds
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print("GPU is available: ", tf.test.is_gpu_available())
camType=0
radioButtonCam=1
FRGraph = FaceRecGraph();
MTCNNGraph = FaceRecGraph();
aligner = AlignCustom();
extract_feature = FaceFeature(FRGraph)
face_detect = MTCNNDetect(MTCNNGraph, scale_factor=2);
class Second(QDialog):
    
    def __init__(self, parent=None):
        super(Second, self).__init__(parent)
        loadUi('addImage.ui',self)
        self.path="dataset\\"
        self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.person_imgs = {"Left" : [], "Right": [], "Center": []};
        self.person_features = {"Left" : [], "Right": [], "Center": []};
        self.init_ui()
        self.count=0
   
    def init_ui(self):
        self.title = "Yüz Ekleme"
        self.top = 200
        self.left = 650
        self.width = 640
        self.height = 640

        imageData=cv2.imread("logo.png",2)
        qformat=QImage.Format_Indexed8
        
        outImage=QImage(imageData,imageData.shape[1],imageData.shape[0],imageData.strides[0],qformat)
        outImage=outImage.rgbSwapped()
        self.labelImage.setPixmap(QPixmap.fromImage(outImage))
        self.labelImage.setScaledContents(True)

        self.setFixedSize(self.width,self.height)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.iconName = "logo.png"
        
        self.lblWarning.setStyleSheet('color: red')
        self.lblWarning.setVisible(False)
#        self.setWindowIcon(QtGui.QIcon(self.iconName))
        self.setWindowTitle(self.title)
        self.image=None
        self.addImage.clicked.connect(self.clickMethod)
    
    def clickMethod(self):
        if self.lineName.text()=="":
            self.lblWarning.setVisible(True)
        else:
            self.lblWarning.setVisible(False)
            detect_name=""
            
            self.capture=cv2.VideoCapture(camType)

            ret,self.image=self.capture.read()
            self.image=cv2.flip(self.image,1)
  
            try:
                self.displayImage(self.image,1)
            except:
                buttonReply = QMessageBox.question(self, 'Uyarı', "Lütfen kamerayı kontrol ediniz.", QMessageBox.Cancel)
                print("orta")
                if buttonReply == QMessageBox.Cancel:

                    firstScreen=FirstScreen()
                    firstScreen.show()
                    self.setVisible(False)
                    self.timer.stop()
            
    
            detect_name=self.detect_person(self.image)            
            if detect_name=="":
                buttonReply = QMessageBox.question(self, 'Uyarı', "Kameranın karşısına geçiniz", QMessageBox.Cancel)
                if buttonReply == QMessageBox.Cancel:
                    second=Second()
                    second.show()
                    self.setVisible(False)
                    self.timer.stop()
            
            
            elif detect_name=="Unknown":
                   
                self.timer=QTimer(self)
                self.timer.timeout.connect(self.update_frame)
                self.timer.start(5)
                
            else:
                buttonReply = QMessageBox.question(self, 'Uyarı', "Yüz Kayıtlı!", QMessageBox.Cancel)
                if buttonReply == QMessageBox.Cancel:
                    second=Second()
                    second.show()
                    self.setVisible(False)
                    self.timer.stop()
    
    
  
    def closeEvent(self, event):
        name=self.lineName.text()
        f = open('./facerec_128D.txt','r');
        data_set = json.loads(f.read());
        for pos in self.person_imgs: #there r some exceptions here, but I'll just leave it as this to keep it simple
            self.person_features[pos] = [np.mean(extract_feature.get_features(self.person_imgs[pos]),axis=0).tolist()]
        
        data_set[name] = self.person_features;
        f = open('./facerec_128D.txt', 'w');
        f.write(json.dumps(data_set))
        
        self.close()
        self.window = Window()
        self.window.show()
        
    def detect_face(self,img):
        
        name=self.lineName.text()        
        f = open('./facerec_128D.txt','r');
        data_set = json.loads(f.read());

        rects, landmarks = face_detect.detect_face(img, 80);  # min face size is set to 80x80
        for (i, rect) in enumerate(rects):
            self.count+=1
            aligned_frame, pos = aligner.align(160,img,landmarks[:,i]);

            if len(aligned_frame) == 160 and len(aligned_frame[0]) == 160:
                self.person_imgs[pos].append(aligned_frame) 
  
                if(self.count>=80):
                    self.count=0
                    self.timer.stop()
                    self.close()

        return img

    def detect_person(self,img):
        person_name=""
        rects, landmarks = face_detect.detect_face(img,80);#min face size is set to 80x80
        aligns = []
        positions = []

        for (i, rect) in enumerate(rects):
            aligned_face, face_pos = aligner.align(160,img,landmarks[:,i])
            if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
                aligns.append(aligned_face)
                positions.append(face_pos)
            else: 
                print("Align face failed") #log        
        if(len(aligns) > 0):
            features_arr = extract_feature.get_features(aligns)
            person_name = self.findPeople(features_arr,positions)
        return person_name
    
    def findPeople(self,features_arr, positions, thres = 0.6, percent_thres = 70):
        f = open('./facerec_128D.txt','r')
        data_set = json.loads(f.read());
        returnRes = ""
        for (i,features_128D) in enumerate(features_arr):
            result = "Unknown";
            smallest = sys.maxsize
            for person in data_set.keys():
                person_data = data_set[person][positions[i]];
                for data in person_data:
                    distance = np.sqrt(np.sum(np.square(data-features_128D)))
                    if(distance < smallest):
                        smallest = distance;
                        result = person;
            percentage =  min(100, 100 * thres / smallest)
            if percentage <= percent_thres :
                result = "Unknown"
            returnRes=result
        return returnRes
    

    def update_frame(self):
        ret,self.image=self.capture.read()
        self.image=cv2.flip(self.image,1)
        try:
            detect_image=self.detect_face(self.image)
        except:
            buttonReply = QMessageBox.question(self, 'Uyarı', "Lütfen kamerayı kontrol ediniz.", QMessageBox.Cancel)
            if buttonReply == QMessageBox.Cancel:
                firstScreen=FirstScreen()
                firstScreen.show()
                self.setVisible(False)
                self.timer.stop()
        
        self.displayImage(detect_image,1)
        
    def displayImage(self,img,window=1):
        qformat=QImage.Format_Indexed8
        if len(img.shape)==3:
            if img.shape[2]==4:
                qformat=QImage.Format_RGBA8888
            else:
                qformat=QImage.Format_RGB888
        outImage=QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)
        outImage=outImage.rgbSwapped()
        
        if window==1:
            self.labelImage.setPixmap(QPixmap.fromImage(outImage))
            self.labelImage.setScaledContents(True)
        if window==2:
            self.processedLabel.setPixmap=(QPixmap.fromImage(outImage))
            self.processedImage.setScaledContents(True)
        
class Window(QtWidgets.QWidget):
    
    def __init__(self):
        super(Window,self).__init__()
        self.detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
        self.path="dataset\\"
        
        
        self.InitWindow()
    
    def InitWindow(self):
        
        self.title = "Yüz Tanıma"
        self.top = 200
        self.left = 650
        self.width = 640
        self.height = 640
        self.setFixedSize(self.width,self.height)
        self.image=None
        self.iconName = "logo.png"
        self.setWindowIcon(QtGui.QIcon(self.iconName))
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.timer=QTimer(self)
        self.run_button = QtWidgets.QPushButton('Yüzü Bul')
        self.addImage = QtWidgets.QPushButton('Veri Ekle')


        self.run_button.clicked.connect(self.findImage)
        self.addImage.clicked.connect(self.imageAdd)
        

        self.vbox = QVBoxLayout()
        self.labelImage=QLabel(self)
        pixmap=QPixmap("logo.png")
        self.labelImage.setPixmap(pixmap)
        self.labelImage.setAlignment(QtCore.Qt.AlignCenter)
        self.vbox.addWidget(self.labelImage)
     
        self.imageBox = QLabel(self)
#        self.p = QPixmap.fromImage(self.p)    
#        self.p = self.p.scaled(640, 480, QtCore.Qt.KeepAspectRatio)
        
#        self.imageBox.resize(1200, 900)
        
        self.vbox.addWidget(self.imageBox)
        self.vbox.addWidget(self.run_button)
        self.vbox.addWidget(self.addImage)

        self.setLayout(self.vbox)
        self.timer.stop()

    def closeEvent(self, event):

        self.close()
        self.timer.stop()
        self.firstScreen=FirstScreen()
        self.firstScreen.show()

#    
    def update_frame(self):
        start_time_fps = time.time()
        ret,self.image=self.capture.read()
        if radioButtonCam==0:
            self.image=cv2.flip(self.image,1)
        
        try:
            detect_image=self.detect_face(self.image)
            
        except:
            buttonReply = QMessageBox.question(self, 'Uyarı', "Lütfen kamerayı kontrol ediniz.", QMessageBox.Cancel)
            if buttonReply == QMessageBox.Cancel:
                self.close()
                self.timer.stop()
       
        YOLO_OBJECT.detection_people( True , self.image)
        cv2.putText(self.image, "FPS : " + str((1.0 / (time.time() - start_time_fps))),
                    (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        self.displayImage(detect_image,1)
    
    def displayImage(self,img,window=1):
        
        qformat=QImage.Format_Indexed8
       
        
        if len(img.shape)==3:
            if img.shape[2]==4:
                qformat=QImage.Format_RGBA8888
            else:
#                
                qformat=QImage.Format_RGB888
                
        outImage=QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)
        outImage=outImage.rgbSwapped()
        if window==1:
            self.imageBox.setPixmap(QPixmap.fromImage(outImage))
#            self.imageBox.setPixmap(p)
            self.imageBox.setScaledContents(True)
        if window==2:
            self.processedLabel.setPixmap=(QPixmap.fromImage(outImage))
            self.processedImage.setScaledContents(True)
    

    
    
    def findImage(self):
        self.labelImage.hide()
        self.capture=cv2.VideoCapture(camType)
#        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,600)
#        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,800)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)
    
    
    def imageAdd(self):
        self.timer.stop()
        self.setVisible(False)
        self.firstScreen=FirstScreen()
        self.firstScreen.setVisible(False)
        self.SW = Second()
        self.SW.show()
        
        
    
    def detect_face(self,img):
        if radioButtonCam==0:
            rects, landmarks = face_detect.detect_face(img,5);
        else:
            rects, landmarks = face_detect.detect_face(img,5);#min face size is set to 80x80
        aligns = []
        positions = []

        for (i, rect) in enumerate(rects):
            aligned_face, face_pos = aligner.align(160,img,landmarks[:,i])
            if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
                aligns.append(aligned_face)
                positions.append(face_pos)
            else: 
                print("Align face failed") #log        
        if(len(aligns) > 0):
            features_arr = extract_feature.get_features(aligns)
            recog_data = self.findPeople(features_arr,positions)
            for (i,rect) in enumerate(rects):
                cv2.rectangle(img,(rect[0],rect[1]),(rect[2],rect[3]),(255,0,0)) #draw bounding box for the face
                cv2.putText(img,recog_data[i][0]+" - "+str(recog_data[i][1])+"%",(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
        return img
    
    def findPeople(self,features_arr, positions, thres = 0.6, percent_thres = 70):
        f = open('./facerec_128D.txt','r')
        data_set = json.loads(f.read());
        returnRes = [];
        for (i,features_128D) in enumerate(features_arr):
            result = "Unknown";
            smallest = sys.maxsize
            for person in data_set.keys():
                person_data = data_set[person][positions[i]];
                for data in person_data:
                    distance = np.sqrt(np.sum(np.square(data-features_128D)))
                    if(distance < smallest):
                        smallest = distance;
                        result = person;
            percentage =  min(100, 100 * thres / smallest)
            if percentage <= percent_thres :
                result = "Unknown"
            returnRes.append((result,percentage))
        return returnRes
#
    
class FirstScreen(QDialog):
    
    def __init__(self, parent=None):
        super(FirstScreen, self).__init__(parent)
        loadUi('firstScreen.ui',self)
        self.init_ui()
        self.count=0

        
    def init_ui(self):
        self.title = "Hoşgeldiniz"
        self.top = 200
        self.left = 650
        self.width = 640
        self.height = 640
        self.setFixedSize(self.width,self.height)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.iconName = "logo.png"
        self.setWindowIcon(QtGui.QIcon(self.iconName))
        self.setWindowTitle(self.title)
        self.nextButton.clicked.connect(self.buttonClick)
        self.rbnLaptop.clicked.connect(self.rbnLaptop_click)
        self.rbnUsb.clicked.connect(self.rbnUsb_click)
        self.rbnHls.clicked.connect(self.rbnHls_click)
        self.rbnRtsp.clicked.connect(self.rbnRtsp_click)
        self.rbnRtmp.clicked.connect(self.rbnRtmp_click)
        self.lineIP.setDisabled(1)
        self.lineUser.setDisabled(1)
        self.linePassword.setDisabled(1)
        self.lblIP.setDisabled(1)
        self.lblUsername.setDisabled(1)
        self.lblPassword.setDisabled(1)
        self.lblWarning.setStyleSheet('color: red')
        self.lblWarning.setVisible(False)
        
    def rbnLaptop_click(self):
    
        self.lineIP.setDisabled(1)
        self.lineUser.setDisabled(1)
        self.linePassword.setDisabled(1)
        self.lblIP.setDisabled(1)
        self.lblUsername.setDisabled(1)
        self.lblPassword.setDisabled(1)
        self.lblWarning.setVisible(False)
    
    def rbnUsb_click(self):
        self.lineIP.setDisabled(1)
        self.lineUser.setDisabled(1)
        self.linePassword.setDisabled(1)
        self.lblIP.setDisabled(1)
        self.lblUsername.setDisabled(1)
        self.lblPassword.setDisabled(1)
        self.lblWarning.setVisible(False)
        
    def rbnHls_click(self):
        self.lineIP.setEnabled(1)
        self.lineUser.setDisabled(1)
        self.linePassword.setDisabled(1)
        self.lblIP.setEnabled(1)
        self.lblUsername.setDisabled(1)
        self.lblPassword.setDisabled(1)
        self.lblWarning.setVisible(False)
        
    def rbnRtsp_click(self):
        self.lineIP.setEnabled(1)
        self.lineUser.setEnabled(1)
        self.linePassword.setEnabled(1)
        self.lblIP.setEnabled(1)
        self.lblUsername.setEnabled(1)
        self.lblPassword.setEnabled(1)
        self.lblWarning.setVisible(False)
        
    def rbnRtmp_click(self):
        self.lineIP.setEnabled(1)
        self.lineUser.setEnabled(1)
        self.linePassword.setEnabled(1)
        self.lblIP.setEnabled(1)
        self.lblUsername.setEnabled(1)
        self.lblPassword.setEnabled(1)
        self.lblWarning.setVisible(False)
        
    def buttonClick(self):
        global camType
        global radioButtonCam
        if(self.rbnLaptop.isChecked()):
            #dizüstü
            camType=0
            radioButtonCam=0
            self.close()
            self.window=Window()
            self.window.show()

        elif(self.rbnUsb.isChecked()):
            #USB
            camType=1
            radioButtonCam=0
            self.window=Window()
            self.window.show()
            self.close()
            
        elif(self.rbnHls.isChecked()):
            #hls            
            radioButtonCam=1
            if(self.lineIP.text()==""):
                self.lblWarning.setVisible(True)
            else:
                
                if(self.lineUser.text()=="" and self.linePassword.text()==""):
                    camType=self.lineIP.text()
                else:                    
                    splitIp=self.lineIP.text().split("://")
                    camType=(splitIp[0]+"://"+self.lineUser.text()+":"+self.linePassword.text()+"@"+splitIp[1]) 

                self.close()
                self.window=Window()
                self.window.show()
        
        elif(self.rbnRtsp.isChecked()):
            #rtsp
            radioButtonCam=1
            if(self.lineIP.text()==""):
                self.lblWarning.setVisible(True)
            else:
                self.lblWarning.setVisible(False)
                if(self.lineUser.text()=="" and self.linePassword.text()==""):
                    camType=self.lineIP.text()
                    print(camType)
                    
                else:
                    splitIp=self.lineIP.text().split("://")
                    camType=(splitIp[0]+"://"+self.lineUser.text()+":"+self.linePassword.text()+"@"+splitIp[1])
                    print(camType)
                self.close()
                self.window=Window()
                self.window.show()
            
        elif(self.rbnRtmp.isChecked()):
            #rtmp
            radioButtonCam=1
            if(self.lineIP.text()==""):
                self.lblWarning.setVisible(True)
            else:
                self.lblWarning.setVisible(False)
                if(self.lineUser.text()=="" and self.linePassword.text()==""):
                    camType=self.lineIP.text()
                else:                    
                    splitIp=self.lineIP.text().split("://")
                    camType=(splitIp[0]+"://"+self.lineUser.text()+":"+self.linePassword.text()+"@"+splitIp[1]) 

                self.close()
                self.window=Window()
                self.window.show()
        else:
              buttonReply = QMessageBox.question(self, 'Uyarı', "Lütfen bir kamera seçiniz!", QMessageBox.Cancel)
  
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_widget = FirstScreen()
    main_widget.show()
    sys.exit(app.exec_())

    