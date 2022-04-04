import tkinter as tk
import cv2
import PIL
from PIL import Image, ImageTk
from matplotlib import image
import datetime
import os
from tkinter import filedialog

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
        
        
        snapshot = tk.Button(self.camera_window, text='Take a snapshot', width=50, font=['Times', '8', 'bold'])
        snapshot.pack(fill='both', padx=10, pady=5, anchor='n')
        snapshot.bind('<Button>', lambda x: self.take_snapshot())

        back = tk.Button(self.camera_window, text='Back', width=50, font=['Times', '8', 'bold'])
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
        
        load_image = tk.Button(self.image_window, text='Load image for detection', width=50, font=['Times', '8', 'bold'])  
        load_image.pack(fill='both', padx=10, pady=5, anchor='n')
        load_image.bind('<Button>', lambda x: self.browse_img_file())
        
        back = tk.Button(self.image_window, text='Back', width=50, font=['Times', '8', 'bold'])
        back.pack(fill='both', padx=10, pady=5, anchor='n')
        back.bind('<Button>', lambda x: self.main_window(self.image_window))

        self.image_window.pack(fill='both', expand=True)

    def draw_found_faces(self, detected, image, color: tuple):
        for (x, y, width, height) in detected:
            cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness=2 )

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

    def take_snapshot(self):
        ts = datetime.datetime.now()
        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
        p = os.path.sep.join((self.outputPath, filename))
        cv2.imwrite(p, self.camera_frame.copy())

    def browse_img_file(self):
        try:
            filename = filedialog.askopenfilename(initialdir = "/", title = "Select an image", filetypes = (("JPG files", "*.JPG*"), ("all files", "*.*")))
            img = Image.open(filename).resize((640, 480), Image.ANTIALIAS)
            self.img = ImageTk.PhotoImage(img)
            self.image_frame.create_image(0, 0, image = self.img, anchor = tk.NW)
            
        except:
            return

        
if __name__ == '__main__':

    root = tk.Tk()
    app = gad_app(root)
    root.geometry('640x640')
    root.title('Gender and Age Detecion Application')
    root.mainloop()