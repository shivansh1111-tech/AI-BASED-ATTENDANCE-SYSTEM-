import cv2
import os

# Create Dataset folder if not exists
if not os.path.exists("Dataset"):
    os.makedirs("Dataset")

user_id = input("Enter User ID (numbers only): ")

cam = cv2.VideoCapture(0 , cv2.CAP_DSHOW)
if not cam.isOpened():
    print("Camera not opening!")
    exit() 
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count = 0

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1

        # Save image inside Dataset folder
        cv2.imwrite(f"Dataset/User.{user_id}.{count}.jpg", gray[y:y+h, x:x+w])

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Capture", img)

    if cv2.waitKey(1) == 27 or count >= 30:
        break

cam.release()
cv2.destroyAllWindows()

print("Images Saved Successfully!")