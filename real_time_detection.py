import tkinter as tk
import cv2
import PIL
from PIL import Image, ImageTk
import datetime
import os
from tkinter import filedialog
import numpy as np
from pathlib import PurePosixPath
from tensorflow.keras.models import  load_model

class gad_app(tk.Frame):
    def __init__(self, master=None):
        super().__init__()
        self.master = master
        self.vid = cv2.VideoCapture(0)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", 0)

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        self.gender_model = load_model('ml_model/gender_model.h5')
        self.age_model = load_model('ml_model/age_model.h5')

        self.main_window()

    def main_window(self, frame = None):
        if frame !=None:
            frame.destroy()

        self.main_window_frame = tk.Frame(width=500, height=500)
        self.main_window_frame
        main_window_title = tk.Label(self.main_window_frame, text='Gender and Age Detection Application', anchor=tk.CENTER, font=['Times', '20', 'bold'])
        main_window_title.pack(padx=10, pady=25)

        real_time_camera = tk.Button(self.main_window_frame, text='Real Time Detection', width=50, height=10, font=['Times', '12', 'bold'])
        real_time_camera.pack(fill='both', padx=10, pady=20, anchor='n')
        real_time_camera.bind('<Button>', lambda x: self.camera_main_window())

        load_image_detection = tk.Button(self.main_window_frame, text='Image Detection', width=50,  height=10, font=['Times', '12', 'bold'])
        load_image_detection.pack(fill='both', padx=10, pady=20, anchor='n')
        load_image_detection.bind('<Button>', lambda x: self.image_main_window())

        self.main_window_frame.pack(fill='both', expand=True)

    def camera_main_window(self):
        self.main_window_frame.destroy()

        self.camera_window = tk.Frame(width=500, height=500)

        app_title = tk.Label(self.camera_window, text='Real Time Gender and Age Detection', anchor=tk.CENTER, font=['Times', '20', 'bold'])
        app_title.pack(padx=10, pady=10)

        self.camera_frame = tk.Canvas(self.camera_window, bd=2, relief="groove", width=self.width, height=self.height)
        self.camera_frame.pack(fill='both', padx=10, pady=10)
        
        back = tk.Button(self.camera_window, text='Back', width=50, font=['Times', '10', 'bold'])
        back.pack(fill='both', padx=10, pady=5, anchor='n')
        back.bind('<Button>', lambda x: self.main_window(self.camera_window))

        self.camera_window.pack(fill='both', expand=True)

        self.start_camera()

    def image_main_window(self):
        self.main_window_frame.destroy()

        self.image_window = tk.Frame(width=500, height=500)

        app_title = tk.Label(self.image_window, text='Load Image Gender and Age Detection', anchor=tk.CENTER, font=['Times', '20', 'bold'])
        app_title.pack(padx=10, pady=10)

        self.image_frame = tk.Canvas(self.image_window, bd=2, relief="groove", width=self.width, height=self.height)
        self.image_frame.pack(fill='both', padx=10, pady=10)
        
        load_image = tk.Button(self.image_window, text='Load image for detection', width=50, font=['Times', '10', 'bold'])  
        load_image.pack(fill='both', padx=10, pady=5, anchor='n')
        load_image.bind('<Button>', lambda x: self.browse_img_file())
        
        back = tk.Button(self.image_window, text='Back', width=50, font=['Times', '10', 'bold'])
        back.pack(fill='both', padx=10, pady=5, anchor='n')
        back.bind('<Button>', lambda x: self.main_window(self.image_window))

        self.image_window.pack(fill='both', expand=True)

    def draw_found_faces(self, detected, image, color: tuple, real=True):
        gen_encoder = {0: 'Female', 1: 'Male'}
        age_encoder = {0: '0-2', 1: '4-6', 2: '8-13', 3: '15-20', 4: '25-32', 5: '38-43', 6: '48-53', 7: '60+'}
        
        if len(detected) > 1:

            for (x, y, width, height) in detected:
                crop_img = image[y: y+height, x: x + width]
                img_to_pred = np.array(cv2.resize(crop_img, (224, 224))).reshape(1, 224, 224, 3)
                gender_pred = self.gender_model.predict(img_to_pred).argmax(1)[0]
                age_pred = self.age_model.predict(img_to_pred).argmax(1)[0]
                gender = gen_encoder[gender_pred]
                age = age_encoder[age_pred]
                prediction_gender = f'Gender: {gender}'
                prediction_age = f'Age: {age}'
                cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness=3)
                
                if real == True:

                    text_scale = min(width, height) * 2e-3
                    text_thickness = int(np.ceil(min(width, height) * 1e-3))
                    cv2.putText(image, prediction_gender,  (x , y - 30), cv2.FONT_HERSHEY_SIMPLEX, text_scale , color, text_thickness)
                    cv2.putText(image, prediction_age,  (x , y - 10), cv2.FONT_HERSHEY_SIMPLEX, text_scale , color, text_thickness)
                
                else:

                    text_scale = 0.7
                    text_thickness = 2
                    cv2.putText(image, prediction_gender,  (x , y - 50), cv2.FONT_HERSHEY_SIMPLEX, text_scale , color, text_thickness)
                    cv2.putText(image, prediction_age,  (x , y - 25), cv2.FONT_HERSHEY_SIMPLEX, text_scale , color, text_thickness)

        else:

            for (x, y, width, height) in detected:
                img_to_pred = np.array(cv2.resize(image, (224, 224))).reshape(1, 224, 224, 3)
                gender_pred = self.gender_model.predict(img_to_pred).argmax(1)[0]
                age_pred = self.age_model.predict(img_to_pred).argmax(1)[0]
                gender = gen_encoder[gender_pred]
                age = age_encoder[age_pred]
                prediction_gender = f'Gender: {gender}'
                prediction_age = f'Age: {age}'
                cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness=3)
                
                if real == True:

                    text_scale = min(width, height) * 2e-3
                    text_thickness = int(np.ceil(min(width, height) * 1e-3))
                    cv2.putText(image, prediction_gender,  (x , y - 30), cv2.FONT_HERSHEY_SIMPLEX, text_scale , color, text_thickness)
                    cv2.putText(image, prediction_age,  (x , y - 10), cv2.FONT_HERSHEY_SIMPLEX, text_scale , color, text_thickness)
                
                else:

                    text_scale = 0.7
                    text_thickness = 2
                    cv2.putText(image, prediction_gender,  (x , y - 50), cv2.FONT_HERSHEY_SIMPLEX, text_scale , color, text_thickness)
                    cv2.putText(image, prediction_age,  (x , y - 25), cv2.FONT_HERSHEY_SIMPLEX, text_scale , color, text_thickness)


    def start_camera(self):
        _, frame = self.vid.read()
        grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = self.face_cascade.detectMultiScale(image=grayscale_image, scaleFactor=1.3, minNeighbors=4)
        
        self.draw_found_faces(detected_faces, frame, (0, 255, 0))
        
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = PIL.Image.fromarray(cv2image)
        self.photo = ImageTk.PhotoImage(image=img)
        self.camera_frame.create_image(0, 0, image = self.photo, anchor = tk.NW)
        self.camera_frame.after(10, lambda: self.start_camera())

    def browse_img_file(self):
        filename = filedialog.askopenfilename(initialdir = "/", title = "Select an image", filetypes = (("JPG files", "*.JPG*"), ("all files", "*.*")))        
        filename = str(PurePosixPath(str(filename)))
        self.img = np.array(Image.open(filename))
        grayscale_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        detected_faces = self.face_cascade.detectMultiScale(image=grayscale_image, scaleFactor=1.3, minNeighbors=3)
        self.draw_found_faces(detected_faces, self.img, (0, 255, 0), real=False)
        self.img = PIL.Image.fromarray(self.img).resize((720, 480))
        self.img = ImageTk.PhotoImage(image=self.img)
        img_width = self.img.width()
        img_height = self.img.height()
        self.image_frame.config(width=img_width, height=img_height)
        self.image_frame.create_image(0, 0, image = self.img, anchor = tk.NW)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    root = tk.Tk()
    app = gad_app(root)
    root.title('Gender and Age Detecion Application')
    root.mainloop()