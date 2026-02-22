from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from io import BytesIO
import base64
import time
import uuid
import uvicorn
import logging

from logging_config import setup_logging

from services.AadharMasks import mask
from services.AutoCrop import crop_image
from services.FaceExtract import face_crop
from services.PanSignature import pansignature
from services.ImageOrientation import image_orientation
from services.Pan_Ocr_Service import extract_text, parse_pan_card
from services.AadharFront_Ocr_Service import extract_text_front, parse_aadhaar_card
from services.AadharBack_Ocr_Service import extract_text_back, parse_aadhaar_card_back
from services.VoterIdOcr import extract_text_voter_front, parse_voter_id


# Setup
app = FastAPI(debug=True)
ALLOWED_ORG_ID = "AlphafinSoft2026"
setup_logging()
logger = logging.getLogger(__name__)


#===================================== MIDDLEWARE =====================================

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Middleware for request validation and logging"""
    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Read headers
    organization_id = request.headers.get("organizationid")
    requester_id = request.headers.get("requesterid")

    # Validate required headers
    if not organization_id:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "organization-id header is required"
            }
        )

    if organization_id != ALLOWED_ORG_ID:
        return JSONResponse(
            status_code=401,
            content={
                "success": False,
                "message": "Invalid organization-id"
            }
        )

    # Store in request state
    request.state.request_id = request_id
    request.state.organization_id = organization_id
    request.state.requester_id = requester_id

    # Log request
    logger.info(
        f"REQUEST [{request_id}] {request.method} {request.url.path} "
        f"| Org: {organization_id} | Req: {requester_id or 'N/A'}"
    )

    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(
            f"ERROR [{request_id}] Unhandled exception: {str(e)} "
            f"| Org: {organization_id} | Req: {requester_id or 'N/A'}"
        )
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Internal server error"
            }
        )

    # Log response
    process_time = round(time.time() - start_time, 3)
    logger.info(
        f"RESPONSE [{request_id}] Status: {response.status_code} | Time: {process_time}s "
        f"| Org: {organization_id} | Req: {requester_id or 'N/A'}"
    )

    # Add response headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Organization-ID"] = organization_id
    if requester_id:
        response.headers["X-Requester-ID"] = requester_id

    return response



#===================================== HELPER FUNCTIONS =====================================


def validate_image_file(image: UploadFile, allowed_extensions: set = None):
    """Validate uploaded image file"""
    if allowed_extensions is None:
        allowed_extensions = {"jpg", "jpeg", "png", "bmp", "tiff"}
    
    if not image.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    ext = image.filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Only image files are allowed")
    
    return ext


async def process_image_endpoint(image: UploadFile, processor_func, endpoint_name: str, 
                               success_message: str = "Processing successful"):
    """Generic image processing endpoint handler"""
    logger.info(f"{endpoint_name} | File: {image.filename}")
    
    # Validate file
    validate_image_file(image)
    
    try:
        # Read and process image
        image_bytes = await image.read()
        logger.info(f"{endpoint_name} | File size: {len(image_bytes)} bytes")
        
        result = processor_func(image_bytes)
        
        if not result:
            logger.warning(f"{endpoint_name} | Processing returned empty result")
            raise HTTPException(status_code=404, detail="No result found")
        
        logger.info(f"{endpoint_name} | {success_message}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"{endpoint_name} | Internal error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# =====================================IMAGE ORIENTATION =====================================


@app.post("/imageOrientation")
async def image_orientation_api(
    request: Request,
    image: UploadFile = File(...)
):
    """
    Correct image orientation
    """

    # ✅ (Optional) Access middleware values if needed for logging
    request_id = getattr(request.state, "request_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    requester_id = getattr(request.state, "requester_id", None)

    # Example log (optional)
    # logger.info(
    #     f"[{request_id}] IMAGE ORIENTATION | Org={organization_id} | Req={requester_id}"
    # )

    # ✅ Read uploaded image bytes
    image_bytes = await image.read()

    # ✅ Convert to BytesIO and call service (NO CHANGE to service)
    return await image_orientation(BytesIO(image_bytes))

# =====================================
# VOter OCR
# =====================================
@app.post("/VoterId/extract")
async def extract_voter_id_front(image: UploadFile = File(...)):
    """
    Extract Voter ID front information
    """

    logger.info(f"VOTER FRONT EXTRACT | File: {image.filename}")

    try:
        # Validate image (your existing function)
        validate_image_file(image)

        # Read image bytes
        image_bytes = await image.read()
        logger.info(
            f"VOTER FRONT EXTRACT | File size: {len(image_bytes)} bytes"
        )

        # OCR extraction
        ocr_text = extract_text_voter_front(image_bytes)
        logger.info(
            f"VOTER FRONT EXTRACT | OCR text length: {len(ocr_text)}"
        )

        # Parse OCR text using Ollama LLM
        result = parse_voter_id(ocr_text)
        logger.info("VOTER FRONT EXTRACT | Parsing successful")

        return JSONResponse(content=result)

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(
            f"VOTER FRONT EXTRACT | Internal error: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

#===================================== PAN OCR =====================================

@app.post("/pan/extract")
async def extract_pan(image: UploadFile = File(...)):
    """Extract PAN card information"""
    logger.info(f"PAN EXTRACT | File: {image.filename}")
    
    validate_image_file(image)
    
    try:
        image_bytes = await image.read()
        logger.info(f"PAN EXTRACT | File size: {len(image_bytes)} bytes")
        
        ocr_text = extract_text(image_bytes).strip()
        logger.info(f"PAN EXTRACT | OCR text length: {len(ocr_text)}")
        
        result = parse_pan_card(ocr_text)
        logger.info("PAN EXTRACT | Extraction successful")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PAN EXTRACT | Internal error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")



#===================================== AADHAAR FRONT OCR =====================================
 
@app.post("/AadharFront/extract")
async def extract_aadhar(image: UploadFile = File(...)):
    """Extract Aadhaar front information"""
    logger.info(f"AADHAAR FRONT EXTRACT | File: {image.filename}")
    
    validate_image_file(image)
    
    try:
        image_bytes = await image.read()
        logger.info(f"AADHAAR FRONT EXTRACT | File size: {len(image_bytes)} bytes")
        
        ocr_text = extract_text_front(image_bytes)
        logger.info(f"AADHAAR FRONT EXTRACT | OCR text length: {len(ocr_text)}")
        
        result = parse_aadhaar_card(ocr_text)
        logger.info("AADHAAR FRONT EXTRACT | Parsing successful")
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AADHAAR FRONT EXTRACT | Internal error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


#===================================== AADHAAR BACK OCR =====================================

@app.post("/AadharBack/extract")
async def extract_aadhar_back(image: UploadFile = File(...)):
    """Extract Aadhaar back information"""
    logger.info(f"AADHAAR BACK EXTRACT | File: {image.filename}")
    
    validate_image_file(image)
    
    try:
        image_bytes = await image.read()
        logger.info(f"AADHAAR BACK EXTRACT | File size: {len(image_bytes)} bytes")
        
        ocr_text = extract_text_back(image_bytes)
        logger.info(f"AADHAAR BACK EXTRACT | OCR text length: {len(ocr_text)}")
        
        result = parse_aadhaar_card_back(ocr_text)
        logger.info("AADHAAR BACK EXTRACT | Parsing successful")
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AADHAAR BACK EXTRACT | Internal error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ===================================== PAN SIGNATURE EXTRACTION =====================================

@app.post("/pansignature")
async def pansignature_api(image: UploadFile = File(...)):
    """Extract signature from PAN card"""
    logger.info(f"PAN SIGNATURE | File: {image.filename}")
    
    validate_image_file(image)
    
    try:
        file_bytes = await image.read()
        logger.info(f"PAN SIGNATURE | File size: {len(file_bytes)} bytes")
        
        cropped_stream = pansignature(file_bytes, image.filename)
        
        if not cropped_stream:
            logger.warning("PAN SIGNATURE | Signature not detected")
            raise HTTPException(status_code=404, detail="Signature not detected")
        
        base64_signature = base64.b64encode(cropped_stream.read()).decode("utf-8")
        logger.info("PAN SIGNATURE | Signature extracted successfully")
        
        return {"signature_base64": base64_signature}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PAN SIGNATURE | Internal error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ===================================== FACE CROP =====================================

@app.post("/facecrop")
async def face_crop_api(image: UploadFile = File(...)):
    """Crop face from image"""
    logger.info(f"FACE CROP | File: {image.filename}")
    
    validate_image_file(image)
    
    try:
        image_bytes = await image.read()
        logger.info(f"FACE CROP | File size: {len(image_bytes)} bytes")
        
        cropped_stream = face_crop(image_bytes)
        
        if not cropped_stream:
            logger.warning("FACE CROP | No face detected")
            raise HTTPException(status_code=404, detail="No face detected")
        
        base64_face = base64.b64encode(cropped_stream.read()).decode("utf-8")
        logger.info("FACE CROP | Face cropped successfully")
        
        return {"face_base64": base64_face}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"FACE CROP | Internal error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ===================================== AADHAAR MASKING =====================================

@app.post("/aadharmask")
async def aadhar_mask_api(image: UploadFile = File(...)):
    """Mask Aadhaar number in image"""
    logger.info(f"AADHAAR MASK | File: {image.filename}")
    
    validate_image_file(image)
    
    try:
        image_bytes = await image.read()
        logger.info(f"AADHAAR MASK | File size: {len(image_bytes)} bytes")
        
        result = mask(image_bytes)
        
        if result is None:
            logger.error("AADHAAR MASK | Masking failed")
            raise HTTPException(status_code=500, detail="Aadhaar masking failed")
        
        # Return Aadhaar number if string
        if isinstance(result, str):
            logger.info("AADHAAR MASK | Aadhaar number extracted")
            return {"aadhaar_number": result}
        
        # Return masked image
        base64_img = base64.b64encode(result.getvalue()).decode()
        logger.info("AADHAAR MASK | Image masked successfully")
        
        return {"base64_img": base64_img}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AADHAAR MASK | Internal error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ===================================== DOCUMENT CROP =====================================

@app.post("/docscrop")
async def autocrop(image: UploadFile = File(...)):
    """Auto-crop document from image"""
    logger.info(f"DOCUMENT CROP | File: {image.filename}")
    
    validate_image_file(image)
    
    try:
        image_bytes = await image.read()
        logger.info(f"DOCUMENT CROP | File size: {len(image_bytes)} bytes")
        
        image_stream = BytesIO(image_bytes)
        cropped_stream, error = crop_image(image_stream)
        
        if error:
            logger.error(f"DOCUMENT CROP | Crop failed: {error}")
            raise HTTPException(status_code=404, detail=error)
        
        base64_crop = base64.b64encode(cropped_stream.read()).decode("utf-8")
        logger.info("DOCUMENT CROP | Image cropped successfully")
        
        return {"image_base64": base64_crop}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"DOCUMENT CROP | Internal error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ===================================== GLOBAL EXCEPTION HANDLER  =====================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions"""
    logger.error(f"UNHANDLED EXCEPTION at {request.url}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"}
    )

# ===================================== MAIN ENTRY POINT =====================================

if __name__ == "__main__":
    logger.info("FastAPI Server Starting on port 5006")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5006,
        reload=True,
        log_level="debug"
    )