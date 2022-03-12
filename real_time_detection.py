import tkinter as tk
import cv2
import PIL
from PIL import Image, ImageTk
from matplotlib import image

class gad_app(tk.Frame):
    def __init__(self, master=None):
        super().__init__()
        self.master = master
        self.vid = cv2.VideoCapture(0)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", 0)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        self.main_window_init()

    def main_window_init(self):
        
        self.main_window = tk.Frame(width=500, height=500)

        app_title = tk.Label(self.main_window, text='Real Time Gender and Age Detection', anchor=tk.CENTER, font=['Times', '20', 'bold'])
        app_title.pack(padx=10, pady=10)

        self.camera_frame = tk.Canvas(self.main_window, bd=2, relief="groove", width=self.width, height=self.height)
        self.camera_frame.pack(fill='both', padx=10, pady=10)

        detect = tk.Button(self.main_window, text='Start', width=50)
        detect.pack(fill='both', padx=10, pady=5, anchor='n')
        detect.bind('<Button>', lambda x: self.start_camera())

        stop = tk.Button(self.main_window, text='Stop', width=50)
        stop.pack(fill='both', padx=10, pady=5, anchor='n')
        # stop.bind('<Button>', lambda x: self.start_camera())

        self.main_window.pack(fill='both', expand=True)

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

if __name__ == '__main__':

    root = tk.Tk()

    app = gad_app(root)
    root.geometry('700x700')
    root.title('Gender and age detecion')

    root.mainloop()