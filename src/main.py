import cv2
import face_recognition
import numpy as np
import json


def load_face_data(file_path="face_encodings.json"):
    with open(file_path, "r") as f:
        data = json.load(f)
    known_faces = [np.array(enc) for enc in data["encodings"]]
    known_names = data["names"]
    return known_faces, known_names


def detect_face_with_camera():
    known_faces, known_names = load_face_data()

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()

        if not ret:
            print("Failed to capture video frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)

        print("Face locations:", face_locations)
        print("RGB frame shape:", rgb_frame.shape)

        if face_locations:
            face_encodings = face_recognition.face_encodings(
                rgb_frame, face_locations)
        else:
            face_encodings = []

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(
                known_faces, face_encoding)
            name = "Tidak Dikenali"

            face_distances = face_recognition.face_distance(
                known_faces, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


detect_face_with_camera()
