from deepface import DeepFace
import numpy as np
import cv2

def postprocess_face(image: np.array):
    
    image = image * 255.0
    image = image.astype(np.uint8)
    
    return image

def preprocess_face(image: np.array):
    
    image = cv2.resize(image, (100, 100)) / 255.0
    image = image.astype("float32")
    
    return image

def detect_face(image: np.array):
    
    try:
        detections = DeepFace.extract_faces(image, 'retinaface')
        faces = [face['face'] for face in detections]
        
        faces = list(map(postprocess_face, faces))
        faces = list(map(preprocess_face, faces))
        
        print("Found 1 face" if len(faces) == 1 else f"Found {len(faces)} faces")
        
    except Exception as e:
        
        faces = "No face detected"
        print(faces)
    
    return faces