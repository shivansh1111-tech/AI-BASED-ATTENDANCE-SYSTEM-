import cv2
import os

# Ask user ID (Admission Number)
user_id = input("Enter your ID (number): ")

# Create dataset folder if not exists
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Load face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start camera (Windows stable version)
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

count = 0

while True:
    ret, img = cam.read()

    if not ret:
        print("Camera not working")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1

        # Save face image
        cv2.imwrite(
            f"dataset/User.{user_id}.{count}.jpg",
            gray[y:y+h, x:x+w]
        )

        # Draw rectangle
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Capture", img)

    # Stop after 30 images or ESC key
    if cv2.waitKey(1) == 27 or count >= 30:
        break

cam.release()
cv2.destroyAllWindows()

print("Dataset collection completed.")