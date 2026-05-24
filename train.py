import cv2
import os
import numpy as np
from PIL import Image

path = "Dataset"

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
ids = []

for image in os.listdir(path):

    # Only allow correct image format
    if image.startswith("User.") and image.endswith(".jpg"):

        image_path = os.path.join(path, image)

        gray_img = Image.open(image_path).convert('L')
        img_numpy = np.array(gray_img, 'uint8')

        parts = image.split('.')

        if len(parts) >= 3 and parts[1].isdigit():
            user_id = int(parts[1])
            faces.append(img_numpy)
            ids.append(user_id)

if len(faces) == 0:
    print("No valid images found in Dataset folder.")
    exit()

recognizer.train(faces, np.array(ids))
recognizer.write("trainer.yml")

print("Training Completed Successfully.")