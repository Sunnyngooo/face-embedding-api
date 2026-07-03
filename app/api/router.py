from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from app.services.face_service import extract_face_vector, compare_face

router = APIRouter()

@router.post("/embeded")
async def get_embedded(file: UploadFile=File(...), model_name: str = Form("Facenet512"), detector_backend: str = Form("retinaface")):
    try: 
        image_bytes = await file.read()
        vector_res = extract_face_vector(image_bytes)
        return {
            "status": "success",
            "model": model_name,
            "detector_backend": detector_backend,
            "filename": file.filename,
            "vector_dimensions": len(vector_res),
            "vector": vector_res
        } 
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    
@router.post("/compare")
async def compare_two_faces(
    file1: UploadFile=File(...),
    file2: UploadFile=File(...),
    model_name: str = Form("Facenet512"),
    detector_backend: str = Form("retinaface")
):
    try:
        img1_bytes = await file1.read()
        img2_bytes = await file2.read()
        
        verification_result = compare_face(img1_bytes, img2_bytes, model_name, detector_backend)
        return {
            "status": "success",
            "model": model_name,
            "detector_backend": detector_backend,
            "is_match": verification_result["verified"],
            "distance": round(verification_result["distance"], 4),
            "threshold": verification_result["threshold"]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")