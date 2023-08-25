from flask import Flask, render_template, request
import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import pyodbc
from PIL import Image

app = Flask(__name__)

face_detector = MTCNN(image_size=160)
face_recognizer = InceptionResnetV1(pretrained='vggface2').eval()

# Database connection
conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=DESKTOP-0CS83SF\SQLEXPRESS;'
                      'Database=face_features;'
                      'UID=hoang_20521346;'
                      'PWD=20521346;')

cursor = conn.cursor()

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Register page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        person_name = request.form.get('person_name')

        cap = cv2.VideoCapture(0)

        count = 0

        while count < 10:
            ret, frame = cap.read()

            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes, _ = face_detector.detect(frame_pil)

            if boxes is not None:
                for box in boxes:
                    x, y, w, h = box.astype(int)
                    x1, y1 = x, y
                    x2, y2 = x + w - 200, y + h - 140

                    face_image = frame[y1:y2, x1:x2]
                    resized_face = cv2.resize(face_image, (160, 160))
                    face_tensor = torch.tensor(resized_face).permute(2, 0, 1).unsqueeze(0).float()
                    face_features = face_recognizer(face_tensor)
                    new_feature_vector = face_features.detach().numpy()

                    cv2.imwrite(f'crawl/{person_name}/face_{count}.jpg', face_image)

                    cursor.execute("INSERT INTO face_features (person_name, feature_vector) VALUES (?, ?)",
                                   (person_name, new_feature_vector.tobytes()))
                    conn.commit()

                    count += 1
                    if count >= 10:
                        break

            cv2.putText(frame, f'Images captured: {count}/10', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Webcam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        return f"Registered and face captured successfully for {person_name} <br><a href='/'>Quay lại</a>"

    return render_template('register.html')

# Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        cap = cv2.VideoCapture(0)
        matching_person = None

        while matching_person is None:
            ret, frame = cap.read()

            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes, _ = face_detector.detect(frame_pil)

            if boxes is not None:
                for box in boxes:
                    x, y, w, h = box.astype(int)
                    x1, y1 = x, y
                    x2, y2 = x + w - 200, y + h - 140

                    face_image = frame[y1:y2, x1:x2]
                    if face_image.size > 0:
                        try:
                            resized_face = cv2.resize(face_image, (160, 160))
                        except cv2.error as e:
                            print("Error while resizing:", e)
                            continue

                        face_tensor = torch.tensor(resized_face).permute(2, 0, 1).unsqueeze(0).float()
                        face_features = face_recognizer(face_tensor)
                        new_feature_vector = face_features.detach().numpy()

                        cursor.execute("SELECT person_name, feature_vector FROM face_features")
                        database_entries = cursor.fetchall()

                        min_distance = float('inf')
                        for entry in database_entries:
                            person_name, database_feature_blob = entry
                            database_feature = np.frombuffer(database_feature_blob, dtype=np.float32)
                            distance = np.linalg.norm(new_feature_vector - database_feature)

                            if distance < min_distance:
                                min_distance = distance
                                matching_person = person_name

                        # Draw bounding box and name
                        cv2.putText(frame, f'Person: {matching_person}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.rectangle(frame, (x, y), (x + w - 200, y + h - 140), (0, 255, 0), 2)

            cv2.imshow('Webcam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        return f"Login successful for {matching_person} <br><a href='/'>Quay lại</a>"

    return render_template('login.html')

if __name__ == '__main__':
    app.run(debug=True)
