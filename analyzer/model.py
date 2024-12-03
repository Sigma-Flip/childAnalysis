from utils import Prompts
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

prompts = Prompts()


class Model:
    def __init__(self):
        """
        OpenAI 클라이언트와 Hugging Face 감정 분석 모델을 초기화하는 클래스.
        """
        self.model_name = 'gpt-4o-mini'
        self.client = self.setClient()
        self.contents = ""  # 입력된 텍스트 데이터 저장
        self.sentiment_model, self.sentiment_tokenizer = self.setSentimentModel()  # Hugging Face 감정 분석 모델 설정
        self.label_map = {
            0: "분노", 1: "툴툴대는", 2: "좌절한", 3: "짜증내는", 4: "방어적인", 5: "악의적인",
            6: "안달하는", 7: "구역질 나는", 8: "노여워하는", 9: "성가신", 10: "슬픔", 11: "실망한",
            12: "비통한", 13: "후회되는", 14: "우울한", 15: "마비된", 16: "염세적인", 17: "눈물이 나는",
            18: "낙담한", 19: "환멸을 느끼는", 20: "불안", 21: "두려운", 22: "스트레스 받는", 23: "취약한",
            24: "혼란스러운", 25: "당혹스러운", 26: "회의적인", 27: "걱정스러운", 28: "조심스러운", 29: "초조한",
            30: "상처", 31: "질투하는", 32: "배신당한", 33: "고립된", 34: "충격 받은", 35: "가난한 불우한",
            36: "희생된", 37: "억울한", 38: "괴로워하는", 39: "버려진", 40: "당황", 41: "고립된(당황한)",
            42: "남의 시선을 의식하는", 43: "외로운", 44: "열등감", 45: "죄책감의", 46: "부끄러운", 47: "혐오스러운",
            48: "한심한", 49: "혼란스러운(당황한)", 50: "기쁨", 51: "감사하는", 52: "신뢰하는", 53: "편안한",
            54: "만족스러운", 55: "흥분", 56: "느긋", 57: "안도", 58: "신이 난", 59: "자신하는"
        }

    @staticmethod
    def setClient():
        """
        OpenAI API 클라이언트를 설정합니다.
        """
        api_key = "OPENAI_API_KEY"  # Replace with your OpenAI API key
        client = OpenAI(api_key=api_key)
        print("Client set up successfully.")
        return client

    @staticmethod
    def setSentimentModel():
        """
        Hugging Face 감정 분석 모델을 설정합니다.
        """
        print("Loading Hugging Face sentiment analysis model...")
        model_name = 'hun3359/klue-bert-base-sentiment'  # 감정 분석 모델 이름
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        print("Hugging Face model loaded successfully.")
        return model, tokenizer

    def updateInfo(self, text: str):
        """
        텍스트 데이터를 업데이트합니다.

        Args:
            text (str): 입력 텍스트 데이터.
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")
        self.contents = text.strip()

    def evaluate_sentiment(self):
        """
        Hugging Face 감정 분석 모델을 사용하여 텍스트의 감정을 분석합니다.

        Returns:
            list[dict]: 감정 분석 결과.
        """
        print("Performing sentiment analysis...")
        lines = self.contents.split("\n")  # 텍스트를 줄 단위로 나눕니다.
        results = []
        for line in lines:
            if not line.strip():  # 빈 줄은 무시
                continue
            inputs = self.sentiment_tokenizer(line, return_tensors='pt', truncation=True, padding=True)
            outputs = self.sentiment_model(**inputs)
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=-1).item()
            results.append({"text": line, "emotion": self.label_map[predicted_class_id]})
        print("Sentiment analysis complete.")
        return results

    def create(self, work: str):
        """
        특정 작업(Summary, Evaluation, Analysis, Timeline, multiEvaluation)을 수행합니다.

        Args:
            work (str): 작업 유형 (Summary, Evaluation, Analysis, Timeline, multiEvaluation).

        Returns:
            str: 작업 결과 텍스트.
        """
        if work not in ['Summary', 'Evaluation', 'Analysis', 'Timeline', 'multiEvaluation']:
            raise KeyError("Invalid work type. Must be one of ['Summary', 'Evaluation', 'Analysis', 'Timeline', 'multiEvaluation']")

        # multiEvaluation 작업인 경우 Hugging Face 모델 사용
        if work == 'multiEvaluation':
            analysis = self.evaluate_sentiment()
            result = "\n".join([f"문장: {line['text']}, 감정: {line['emotion']}" for line in analysis])
            return result

        # Prompt 가져오기
        prompt = prompts.getPrompt(work)

        # Timeline 작업인 경우
        if work == 'Timeline':
            lines = self.contents.split("\n")  # 줄 단위로 타임라인 생성
            timeline_content = "\n".join(f"{i+1}. {line}" for i, line in enumerate(lines))
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"{prompt}\n\n일기 내용:\n{timeline_content}",
                    }
                ],
                model=self.model_name
            )
        else:
            # Summary, Evaluation, Analysis 작업
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"{prompt}\n\n일기 내용:\n{self.contents}",
                    }
                ],
                model=self.model_name
            )

        # 결과 반환
        return chat_completion.choices[0].message.content
    
    
if __name__ == "__main__":
    text = """
오늘은 정말 재미있는 꿈을 꿨어요.
하늘에 커다란 풍선들이 떠다니고 있었어요.
그리고 그 풍선을 타고 친구들이랑 모험을 떠났어요.
"""

model = Model()
model.updateInfo(text)
result = model.create("multiEvaluation")
print(result)

