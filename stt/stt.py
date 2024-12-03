import whisper
import os

def transcribe_audio(audio_path: str, output_file: str = None) -> str:
    """
    주어진 오디오 파일을 Whisper 모델을 사용하여 텍스트로 변환하고, 변환된 텍스트를 저장 및 반환합니다.

    Args:
        audio_path (str): 변환할 오디오 파일의 경로.
        output_file (str, optional): 변환된 텍스트를 저장할 파일 경로. 지정하지 않으면 저장하지 않습니다.

    Returns:
        str: 변환된 텍스트.
    """
    # 모델 로드
    model = whisper.load_model("base")
    
    # 오디오 파일을 텍스트로 변환
    result = model.transcribe(audio_path)
    
    # 변환된 텍스트 가져오기
    transcribed_text = result["text"]
    
    # 텍스트 저장 (파일 경로가 지정된 경우)
    if output_file:
        # 파일 경로에서 디렉토리 추출
        output_dir = os.path.dirname(output_file)
        
        # 디렉토리가 없는 경우 생성
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 텍스트 저장
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcribed_text)
    
    # 변환된 텍스트 반환
    return transcribed_text

# 함수 사용 예시
if __name__ == "__main__":
    audio_path = r"C:\Users\MLLAB\Desktop\anal\example.mp3"
    output_file = r"C:\Users\MLLAB\Desktop\anal\stt\output\transcription.txt"  # 경로 예시
    
    # 변환 및 저장
    transcribed_text = transcribe_audio(audio_path, output_file)
    
    # 결과 출력
    print("Transcribed Text:", transcribed_text)
