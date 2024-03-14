import cv2
import os

data_directory = "data/Face"
os.makedirs(data_directory, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

camera = cv2.VideoCapture(0)

person_name = input("Nhập tên của bạn: ")

while True:
    ret, frame = camera.read()

    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_image = frame[y:y + h, x:x + w]
        file_name = os.path.join(data_directory, f"{person_name}_{len(os.listdir(data_directory)) + 1}.jpg")
        cv2.imwrite(file_name, face_image)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
