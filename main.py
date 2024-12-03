from analyzer import analysisModel
from stt.stt import transcribe_audio

import datetime
file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_analysis")

model = analysisModel()

if __name__ == "__main__":
    '------------------------------config------------------------------------'
    audiofile = r'example.mp3' # 오디오 파일 경로
    stt_output = r'outputs\transcription.txt' # 텍스트파일 저장할 경로
    text = transcribe_audio(audiofile)
    model.updateInfo(text)
    print(text)

    sentiment_result = model.evaluate_sentiment_with_score()
    print(sentiment_result)
    model.visualize_sentiment(sentiment_result, file_name)

    
    
    
    
    
    '---------------------------------- stt -------------------------------'
    
   