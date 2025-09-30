import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

# Path where your reference images are stored
REFERENCE_PATH = "signals/"

# ORB detector for feature matching
orb = cv2.ORB_create()

# Load reference images and compute descriptors
reference_signs = {}

for folder in os.listdir(REFERENCE_PATH):
    folder_path = os.path.join(REFERENCE_PATH, folder)
    if os.path.isdir(folder_path):
        descriptors = []
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path, 0)  # grayscale
            if img is None:
                continue
            kp, des = orb.detectAndCompute(img, None)
            if des is not None:
                descriptors.append(des)
        if descriptors:
            reference_signs[folder] = descriptors

# Function to recognize traffic sign
def recognize_sign(img_path):
    img = cv2.imread(img_path, 0)
    if img is None:
        return "Invalid Image"
    kp, des = orb.detectAndCompute(img, None)
    if des is None:
        return "No features found"

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    best_match = None
    max_matches = 0

    for sign_name, ref_desc_list in reference_signs.items():
        for ref_des in ref_desc_list:
            matches = bf.match(des, ref_des)
            if len(matches) > max_matches:
                max_matches = len(matches)
                best_match = sign_name

    return best_match if best_match else "Unknown"

# ---------- Tkinter UI ----------
root = tk.Tk()
root.title("Traffic Sign Recognition")
root.geometry("600x500")
root.configure(bg="#f2f2f2")

label = Label(root, text="Upload a traffic sign image", font=("Arial", 16), bg="#f2f2f2")
label.pack(pady=20)

img_label = Label(root, bg="#f2f2f2")
img_label.pack(pady=20)

result_label = Label(root, text="", font=("Arial", 18, "bold"), fg="blue", bg="#f2f2f2")
result_label.pack(pady=20)

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if file_path:
        # Show uploaded image
        img = Image.open(file_path)
        img = img.resize((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        img_label.configure(image=img_tk)
        img_label.image = img_tk

        # Recognize sign
        result = recognize_sign(file_path)
        result_label.config(text=f"Recognized: {result}")

upload_btn = Button(root, text="Upload Image", command=upload_image, font=("Arial", 14), bg="#4CAF50", fg="white")
upload_btn.pack(pady=10)

root.mainloop()

