from fastapi import APIRouter, Depends, HTTPException,File, UploadFile
from fastapi.responses import FileResponse
import shutil
# from sqlalchemy.orm import Session
from database import get_db
# from starlette import status

router = APIRouter(
    prefix='/api/anal',  # 경로 프리픽스
)

@router.post("/upload-audio")
async def upload_audio(audio: UploadFile = File(...)):  # 'audio' 필드로 파일 받기
    try:
        print(f"Received file: {audio.filename}")  # 파일 이름 출력

        # 받은 오디오 파일을 저장
        with open("uploaded_audio.wav", "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        
        return {"filename": audio.filename, "message": "File uploaded successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error uploading file: {str(e)}")