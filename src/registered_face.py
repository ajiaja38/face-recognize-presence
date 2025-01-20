import os
import json
import face_recognition
import numpy as np

IMAGE_FOLDER = "./src/data"
OUTPUT_FILE = "face_encodings.json"


def register_faces_hardcoded():
    name = "Tedi"
    image_files = [
        "tedi1.jpeg",
        "tedi2.jpeg",
        "tedi3.jpeg",
        "tedi4.jpeg",
    ]

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            face_data = json.load(f)
            known_faces = [np.array(enc) for enc in face_data["encodings"]]
            known_names = face_data["names"]
    else:
        known_faces = []
        known_names = []

    for image_file in image_files:
        image_path = os.path.join(IMAGE_FOLDER, image_file)

        if os.path.exists(image_path):
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_faces.append(encodings[0])
                known_names.append(name)
        else:
            print(f"File gambar tidak ditemukan: {image_path}")

    face_data = {
        "encodings": [enc.tolist() for enc in known_faces],
        "names": known_names
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(face_data, f)

    print(f"Data pengenalan wajah disimpan ke {OUTPUT_FILE}")


register_faces_hardcoded()
