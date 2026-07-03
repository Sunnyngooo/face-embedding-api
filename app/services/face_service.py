import os 
import cv2
import numpy as np 
from deepface import DeepFace

def preload_models():
    print("Loading model...")
    DeepFace.build_model("Facenet512")
    DeepFace.build_model("ArcFace") 
    DeepFace.build_model("RetinaFace") 
    print("Pre-load done")

def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_cv2 = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    if img_cv2 is None:
        raise ValueError("Uploaded file is invalid")
    return img_cv2

def extract_face_vector(image_bytes: bytes, model_name: str = "Facenet512", detector_backend: str="retinaface")-> list:
    img_cv2 = bytes_to_cv2(image_bytes)
    
    embedding_obj = DeepFace.represent(
        img_path=img_cv2,
        model_name=model_name,
        detector_backend=detector_backend, #"mtcnn" for CPU, retinaface
        align=True,
        enforce_detection=True
    )
    
    return embedding_obj[0]["embedding"]
     
def compare_face(img1_bytes: bytes, img2_bytes: bytes, model_name: str = "Facenet512", detector_backend: str ="retinaface") -> dict:
    img1_cv2 = bytes_to_cv2(img1_bytes)
    img2_cv2 = bytes_to_cv2(img2_bytes)
    
    result = DeepFace.verify(
        img1_path=img1_cv2,
        img2_path=img2_cv2,
        model_name=model_name,
        detector_backend=detector_backend,
        align=True,
        enforce_detection=True
    )
    return result