import os
import sys

sys.path.append(os.path.dirname(__file__) + "/../")

from scipy.misc import imread

from config import load_config
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input

import cv2
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets


cfg = load_config("demo/pose_cfg.yaml")

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)

class Fenetre(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.view = QLabel("")
        self.view.setScaledContents(True);
        self.button = QPushButton('Stop')        #Cration des boutons de l'interface
        self.button.clicked.connect(stop)
        self.button2 = QPushButton('Play')
        self.button2.clicked.connect(play)
        self.keyPressEvent = lambda event: key(self, event)
        
        self.view.setStyleSheet("background-color:black;")
        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addWidget(self.button2)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def display_image(self,img):
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.view.setPixmap(QPixmap.fromImage(qImg))
        self.view.update()
        QApplication.processEvents()

def key(sim, event):
    """commande des boutons avec le clavier"""
    txt = event.text()
    if txt in "qQ":
        app.close()
    elif txt in "pP":
        play()
    elif txt in "sS" :
        stop()
  

def play():

 
    global VIDEO
    
    VIDEO = True
    
    cap = cv2.VideoCapture(0)      #lancement de la caméra
    
    
    cap.set(3,160)      #redimensionnement de la video (reduire la taille des données)
    cap.set(4,120)

    
    while VIDEO :
       
        # Capture frame-by-frame
        ret, image = cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_batch = data_to_input(image)

        # Compute prediction with the CNN
        outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
        scmap, locref, _ = predict.extract_cnn_output(outputs_np, cfg)

        # print (scmap)
        # Extract maximum scoring location from the heatmap, assume 1 person
        pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)

        # Visualise
        #visualize.show_heatmaps(cfg, image, scmap, pose)
        visim = visualize.visualize_joints(image, pose)
        #visim = image 

        fenetre_fft.display_image(visim)
        print("OK")


    cv2.VideoCapture(0).release()     #extinction de la camera
    

     
    
def stop():
    global VIDEO
    
    VIDEO = False
   

if __name__ == "__main__":

    
    
    app = QtWidgets.QApplication(sys.argv)
    fenetre_fft = Fenetre()
    fenetre_fft.resize(700, 900)
    fenetre_fft.show()
    
    app.exec_()

cap = cv2.VideoCapture(0)

