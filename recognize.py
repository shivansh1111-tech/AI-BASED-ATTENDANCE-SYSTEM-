import cv2
import numpy as np
import csv
import os
from datetime import datetime

print("Starting Face Recognition Attendance System...")

# ------------------------------
# Student ID → Name Dictionary
# ------------------------------
student_dict = {
    850: "Shivansh Awasthi"
}

# ------------------------------
# Load Trainer
# ------------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# ------------------------------
# Load Face Cascade
# ------------------------------
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

attendance_file = "attendance.csv"

# ------------------------------
# Create Attendance File if Not Exists
# ------------------------------
if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Name", "Date", "Time"])

print("Press ESC to exit.")

# ------------------------------
# Recognition Loop
# ------------------------------
while True:
    ret, img = cam.read()
    if not ret:
        print("Camera error!")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 60:
            name = student_dict.get(id, "Unknown")

            now = datetime.now()
            date = now.strftime("%d-%m-%Y")
            time = now.strftime("%H:%M:%S")

            # Read existing attendance
            with open(attendance_file, "r") as f:
                data = f.readlines()

            entry_check = f"{id},{name},{date}"

            # Save only once per day
            if not any(entry_check in line for line in data):
                with open(attendance_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([id, name, date, time])
                print("Attendance Marked for", name)

            label = f"{id} - {name}"
            cv2.putText(img, label, (x, y-10), font, 0.8, (0,255,0), 2)

        else:
            cv2.putText(img, "Unknown", (x, y-10), font, 0.8, (0,0,255), 2)

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("Face Recognition Attendance", img)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()