# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QCursor, QIcon, QImage, QPixmap
from PIL import Image, ImageOps
from tkinter import filedialog 
import tensorflow as tf
import numpy as np
import cv2
import tkinter as tk
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox
import sys
import cv2
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model

root = tk.Tk()
root.withdraw()




class Ui_Form(object):
    class_name = ['00000','BKHCM','DHQG','HUFLIT','UEB','USSH']

    def get_model():
        model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

        for layer in model_vgg16_conv.layers:
            layer.trainable = False

        input = Input(shape=(128, 128, 3), name='image_input')
        output_vgg16_conv = model_vgg16_conv(input)

        x = Flatten(name='flatten')(output_vgg16_conv)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dropout(0.5)(x)
        x = Dense(6, activation='softmax', name='predictions')(x)

        my_model = Model(inputs=input, outputs=x)
        my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return my_model

    model = get_model()
    model.load_weights("weights-14-1.00.hdf5")


    def getImage(self):
        nameimg = filedialog.askopenfilename(initialdir="/Image_Cam/", title="Select Image File",
                                              filetypes=(("All Files", "*.*"), ("JPG File", "*.jpg"), ("PNG File", "*.png")))
        if nameimg: 
            try:
                image = Image.open(nameimg).convert("RGB")
                self.label_4.setGeometry(QtCore.QRect(80, 80, 448, 448))
                
                pixmap = QtGui.QPixmap(nameimg)
                self.label_4.setPixmap(pixmap)
                self.label_4.setScaledContents(True)
                
                self.predict(image)
            except Exception as e:
                print(f"Error: {e}")

    def showDia(self):
        dialog =QMessageBox(Form)
        dialog.setWindowTitle("Ban co muon su dung video ?")


    def predict(self,image):
        data = np.ndarray(shape=(1, 128, 128, 3), dtype=np.float32)
        size = (128, 128)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array

        predict = self.model.predict(data)
        title = self.class_name[np.argmax(predict[0])]
        print(title)
        accuracy = np.max(predict[0])
        pred_lbl = f"{title} - {accuracy*100:.2f}%"
        self.label_3.setText(pred_lbl)

    def displayImage(self, img,window=1):
        img=cv2.resize(img,(340,224))
        qformat=QImage.Format_Indexed8
        if len(img.shape)==3:
            if (img.shape[2])==4:
                qformat =QImage.Format_RGBA8888
            else:
                qformat=QImage.Format_RGB888
        img = QImage(img,img.shape[1],img.shape[0],qformat)
        img =img.rgbSwapped()
        self.label_4.setPixmap(QPixmap.fromImage(img))
        self.label_4.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def video(self):
            self.label_4.setGeometry(QtCore.QRect(80, 80, 448, 448))
            self.label_4.setStyleSheet("border-image:url(GUI/Image/UNIVERSITY.png)")

            cam = cv2.VideoCapture(0)

            while(True):
                ret, image_org = cam.read()
                if not ret:
                    continue
                image_org = cv2.resize(image_org, dsize=None,fx=0.5,fy=0.5)

                image = image_org.copy()
                image = cv2.resize(image, dsize=(128, 128))
                image = image.astype('float')*1./255
                image = np.expand_dims(image, axis=0)

                predict = self.model.predict(image)
                print("This picture is: ", self.class_name[np.argmax(predict[0])], (predict[0]))
                print(np.max(predict[0],axis=0))
                if (np.max(predict)>=0.8) and (np.argmax(predict[0])!=0):
                    self.label_3.setText(self.class_name[np.argmax(predict)])


                cv2.imshow("Picture", image_org)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cam.release()
            cv2.destroyAllWindows()

    def make_data(self):
        dir_name = filedialog.askdirectory(initialdir="/data/", title="Select Directory")
    # Code chụp hình từ camera và lưu vào thư mục dữ liệu huấn luyện
        cap = cv2.VideoCapture(0)
        i = 0
        while True:
            i += 1
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.resize(frame, dsize=None, fx=0.3, fy=0.3)
            cv2.imshow('frame', frame)
            if i >= 60:
                print("Số ảnh capture = ", i - 60)
                label = self.comboBox.currentText()
                cv2.imwrite(os.path.join(dir_name, label, f"{i}.png"), frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1200, 800)
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(0, 0, 1200, 800))
        self.label.setStyleSheet("border-image:url(GUI/Image/2.png)")
        self.label.setText("")
        self.label.setObjectName("label")

        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(720, 420, 362, 82))
        self.pushButton.setStyleSheet("QPushButton#pushButton {border-image:url(GUI/Image/Group 1.png);}QPushButton#pushButton:hover {border-image:url(GUI/Image/Group 1 (1).png);}")
        self.pushButton.setText("")
        self.pushButton.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton.clicked.connect(self.getImage)
        self.pushButton.setObjectName("pushButton")

        self.pushButton_3 = QtWidgets.QPushButton(Form)
        self.pushButton_3.setGeometry(QtCore.QRect(720, 540, 362, 82))
        self.pushButton_3.setStyleSheet("QPushButton#pushButton_3 {border-image:url(GUI/Image/Group 2 (3).png);}QPushButton#pushButton_3:hover {border-image:url(GUI/Image/Group 2 (2).png);}")
        self.pushButton_3.setText("")
        self.pushButton_3.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_3.clicked.connect(self.video)
        self.pushButton_3.setObjectName("pushButton_3")

        self.pushButton_4 = QtWidgets.QPushButton(Form)
        self.pushButton_4.setGeometry(QtCore.QRect(720, 660, 362, 82))
        self.pushButton_4.setStyleSheet("QPushButton#pushButton_4 {border-image:url(GUI/Image/make_data_hover.png);}QPushButton#pushButton_4:hover {border-image:url(GUI/Image/make_data.png);}")
        self.pushButton_4.setText("")
        self.pushButton_4.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_4.clicked.connect(self.make_data)
        self.pushButton_4.setObjectName("pushButton_4")

        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(20, 580, 622, 82))
        self.label_2.setStyleSheet("border-image:url(GUI/Image/Group 4.png)")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")

        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(260, 570, 382, 80))
        self.label_3.setStyleSheet("color:rgb(0, 0, 0); font: 87 18pt \"Segoe UI Black\";")
        self.label_3.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")


        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setGeometry(QtCore.QRect(80, 80, 448, 448))
        self.label_4.setStyleSheet("border-image:url(GUI/Image/UNIVERSITY.png)")
        self.label_4.setObjectName("label_4")
        
        self.comboBox = QtWidgets.QComboBox(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))


# import res_rc


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
    label = QtWidgets.QFileDialog.getExistingDirectory(Form, "Select Label Directory", options=QtWidgets.QFileDialog.ShowDirsOnly)
    