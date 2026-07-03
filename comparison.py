import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import time 
from deepface import DeepFace

MODELS = ["Facenet512", "ArcFace"]
DETECTOR = "mtcnn"

ORIGIN_IMG = "Akbar_002.jpg"
SAME_PER_IMG = "Akbar_003.jpg"
DIFF_PER_IMG = "Akbar_Al_Baker_0001.jpg"

def run_comparison():
    print(f"{'='*60}")
    print("Report compare face embedding models")
    print(f"{'='*60}\n")
    
    for model in MODELS:
        print(f"Testing model {model}")
        print("="*60)
        
        dist_same = 0.0
        dist_diff = 0.0
        
        print("1. Testing images of one person")
        start_time = time.time()
        try:
            res_same = DeepFace.verify(
                img1_path=ORIGIN_IMG,
                img2_path=SAME_PER_IMG,
                model_name=model,
                detector_backend=DETECTOR, enforce_detection=False
            )
            time_same = time.time() - start_time
            dist_same = res_same["distance"]
            threshold = res_same["threshold"]
            
            print(f"   - Speed: {time_same:.2f} seconds")
            print(f"   - Distance: {dist_same:.4f} seconds (Thresholds: {threshold})")
            if res_same["verified"]:
                print(f"   - Conclusion: CORRECT!")
            else:
                print(f"   - Conclusion: INCORRECT!")
        except Exception as e:
            print(f"   - Error: {str(e)}")
            
        print("2. Testing images of two different person")
        start_time = time.time()
        try:
            res_diff = DeepFace.verify(
                img1_path=ORIGIN_IMG,
                img2_path=DIFF_PER_IMG,
                model_name=model,
                detector_backend=DETECTOR, enforce_detection=False
            )
            time_diff = time.time() - start_time
            dist_diff = res_diff["distance"]
            threshold = res_diff["threshold"]
            
            print(f"   - Speed: {time_diff:.2f} seconds")
            print(f"   - Distance: {dist_diff:.4f} seconds (Thresholds: {threshold})")
            if res_diff["verified"]:
                print(f"   - Conclusion: CORRECT!")
            else:
                print(f"   - Conclusion: INCORRECT!")
        except Exception as e:
            print(f"   - Error: {str(e)}")
        
        if dist_same > 0 and dist_diff > 0 :
            margin = dist_diff - dist_same
            print(f"\n=> MARGIN: {margin:.4f}")
            print("    As large as the margin as smart as the model")
        print(f"{'='*60}\n")
        
if __name__ ==  '__main__':
    print('Loading models, please wait...')
    DeepFace.build_model("Facenet512")
    DeepFace.build_model("ArcFace")
    print("Downloading is done. Get started with comparing ... \n")
            
    run_comparison()