from anlyzer.wrapper import LLMWrapper
from stt.stt import transcribe_audio
from visualizer.utils import visualize_wordcloud, visualize_sentiment_analysis

import datetime



if __name__ == "__main__":
    '------------------------------config------------------------------------'
    audiofile = r'example.mp3' # 오디오 파일 경로
    stt_output = r'outputs\transcription.txt' # 텍스트파일 저장할 경로
    text = transcribe_audio(audiofile)
    
    
    
    
    '---------------------------------- stt -------------------------------'
    
   