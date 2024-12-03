from .utils import Prompts
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

prompts = Prompts()


import os
import re
import platform
import torch
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI


import os
import re
import platform
import torch
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI


class Model:
    def __init__(self):
        """
        OpenAI 클라이언트와 Hugging Face 감정 분석 모델을 초기화하는 클래스.
        """
        self.model_name = 'gpt-4o-mini'
        self.client = self.setClient()
        self.contents = ""  # 입력된 텍스트 데이터 저장
        self.sentiment_model, self.sentiment_tokenizer = self.setSentimentModel()  # 감정 분류 모델 설정
        self.score_model, self.score_tokenizer = self.setScoreModel()  # 감정 점수 계산 모델 설정
        self.prompts = Prompts()  # 프롬프트 객체
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
        Hugging Face 감정 분류 모델을 설정합니다.
        """
        print("Loading Hugging Face sentiment analysis model...")
        model_name = 'hun3359/klue-bert-base-sentiment'  # 감정 분석 모델 이름
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        print("Hugging Face sentiment analysis model loaded successfully.")
        return model, tokenizer

    @staticmethod
    def setScoreModel():
        """
        Hugging Face 감정 점수 계산 모델을 설정합니다.
        """
        print("Loading Hugging Face score model...")
        model_name = 'cardiffnlp/twitter-roberta-base-sentiment'  # 예시 모델 이름
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        print("Hugging Face score model loaded successfully.")
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

    def evaluate_sentiment_with_score(self):
        """
        감정 분석과 함께 점수를 계산합니다.

        Returns:
            list[dict]: 각 문장에 대한 감정 레이블 및 점수.
        """
        print("Performing sentiment and score analysis...")
        sentences = [sentence.strip() for sentence in self.contents.split('.') if sentence.strip()]
        results = []
        for sentence in sentences:
            # 감정 분류
            inputs = self.sentiment_tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
            outputs = self.sentiment_model(**inputs)
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=-1).item()
            emotion = self.label_map[predicted_class_id]

            # 감정 점수 계산
            score_inputs = self.score_tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
            score_outputs = self.score_model(**score_inputs)
            score_logits = score_outputs.logits
            positive_score = torch.softmax(score_logits, dim=-1)[0, 2].item() * 100  # 긍정 점수 (예시)

            results.append({"text": sentence, "emotion": emotion, "score": positive_score})
        print("Sentiment and score analysis complete.")
        return results

    def create(self, work: str):
        """
        특정 작업(Summary, Evaluation, Analysis, Timeline, multiEvaluation)을 수행합니다.

        Args:
            work (str): 작업 유형 (Summary, Evaluation, Analysis, Timeline, multiEvaluation).

        Returns:
            str: 작업 결과 텍스트.
        """
        valid_works = [
            'Summary', 'Evaluation', 'Analysis', 'Timeline',
            'multiEvaluation', 'Sentiment', 'Comment', 'ActionPlan'
        ]

        if work not in valid_works:
            raise KeyError(f"Invalid work type. Must be one of {valid_works}")

        # multiEvaluation 작업인 경우
        if work == 'multiEvaluation':
            analysis = self.evaluate_sentiment_with_score()
            result = "\n".join(
                [f"문장: {line['text']}, 감정: {line['emotion']}, 점수: {line['score']:.2f}" for line in analysis]
            )
            return result

        # Prompt 가져오기
        prompt = self.prompts.getPrompt(work)

        # OpenAI 작업 수행
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


    def visualize_sentiment(self, results, file_name, output_dir="results"):
        """
        감정 분석 결과를 바탕으로 시각화를 수행합니다.

        Args:
            results (list[dict]): 감정 분석 결과.
            file_name (str): 결과 파일을 저장할 서브 디렉토리 이름.
            output_dir (str): 기본 결과 디렉토리.
        """
        if platform.system() == 'Windows':
            plt.rc('font', family='Malgun Gothic')  # Windows
        elif platform.system() == 'Darwin':  # macOS
            plt.rc('font', family='Apple SD Gothic Neo')  # macOS
        else:
            plt.rc('font', family='NanumGothic')  # Linux (Nanum 폰트 사용)

        # 음수 표시 문제 해결
        plt.rcParams['axes.unicode_minus'] = False

        # 디렉토리 생성
        file_output_dir = os.path.join(output_dir, file_name)
        if not os.path.exists(file_output_dir):
            os.makedirs(file_output_dir)
            print(f"Output directory '{file_output_dir}' created.")

        # 1. 감정 분포 바 플롯 저장
        emotions = [res['emotion'] for res in results]
        emotion_counts = Counter(emotions)
        
        plt.figure(figsize=(12, 6))
        plt.bar(emotion_counts.keys(), emotion_counts.values())
        plt.title('Emotion Distribution')
        plt.xlabel('Emotion')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        bar_plot_path = os.path.join(file_output_dir, "emotion_distribution.png")
        plt.savefig(bar_plot_path)
        plt.close()
        print(f"Emotion distribution bar plot saved as '{bar_plot_path}'.")

        # 2. 감정 통계표 저장
        emotion_df = pd.DataFrame(emotion_counts.items(), columns=['Emotion', 'Count'])
        emotion_df.sort_values(by='Count', ascending=False, inplace=True)
        stats_path = os.path.join(file_output_dir, "emotion_statistics.csv")
        emotion_df.to_csv(stats_path, index=False)
        print(f"Emotion statistics saved as '{stats_path}'.")

        # 3. 감정 변화 추세 저장
        indices = list(range(1, len(results) + 1))
        scores = [res['score'] for res in results if 'score' in res]  # 점수가 있을 경우만
        plt.figure(figsize=(12, 6))
        plt.plot(indices, scores, marker='o', linestyle='-')
        plt.title('Emotion Score Trends')
        plt.xlabel('Sentence Index')
        plt.ylabel('Emotion Score')
        plt.xticks(indices)
        plt.grid()
        plt.tight_layout()
        trends_plot_path = os.path.join(file_output_dir, "emotion_score_trends.png")
        plt.savefig(trends_plot_path)
        plt.close()
        print(f"Emotion score trends plot saved as '{trends_plot_path}'.")

        # 4. 주요 키워드 3개 추출 후 저장
        all_text = " ".join([res['text'] for res in results])
        keywords = self.extract_keywords(all_text)
        keywords_path = os.path.join(file_output_dir, "top_keywords.txt")
        with open(keywords_path, "w") as f:
            f.write("\n".join(keywords[:3]))
        print(f"Top 3 keywords saved as '{keywords_path}'.")

        # 5. 워드 클라우드 저장
        font_path = "/System/Library/Fonts/Supplemental/AppleSDGothicNeo.ttc"  # 적절한 폰트 경로 설정
        wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=font_path).generate(all_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title("Word Cloud")
        wordcloud_path = os.path.join(file_output_dir, "word_cloud.png")
        plt.savefig(wordcloud_path)
        plt.close()
        print(f"Word cloud saved as '{wordcloud_path}'.")

        # 6. 전체 텍스트와 밑줄 시각화 저장
        colors = plt.cm.Reds  # 감정 점수에 따라 색상을 조정하기 위한 컬러맵
        max_score = max(scores) if scores else 1
        min_score = min(scores) if scores else 0

        plt.figure(figsize=(12, len(results) * 0.5))
        y_offset = 1  # 텍스트 출력의 Y 위치 초기값

        for res in results:
            text = res['text']
            score = res.get('score', 0)  # 점수가 없는 경우 0으로 처리
            score_normalized = (score - min_score) / (max_score - min_score) if scores else 0.5
            color = colors(score_normalized)

            # 텍스트 출력
            plt.text(0, y_offset, text, fontsize=10, va='center', ha='left')

            # 텍스트 길이에 맞춰 밑줄 그리기
            text_length = len(text) * 0.12  # 텍스트 길이를 동적으로 계산
            plt.plot([0, text_length], [y_offset - 0.1, y_offset - 0.1], color=color, lw=4)

            y_offset -= 0.5  # 다음 줄로 이동

        plt.axis('off')  # 축 숨김
        underline_plot_path = os.path.join(file_output_dir, "text_with_underline.png")
        plt.savefig(underline_plot_path)
        plt.close()
        print(f"Text with underlines saved as '{underline_plot_path}'.")


    @staticmethod
    def extract_keywords(text):
        """
        텍스트에서 주요 키워드를 추출합니다.

        Args:
            text (str): 입력 텍스트.

        Returns:
            list[str]: 키워드 목록.
        """
        words = re.findall(r'\b\w+\b', text.lower())
        word_counts = Counter(words)
        return [word for word, _ in word_counts.most_common()]


if __name__ == "__main__":
    text = """
    오늘은 정말 재미있는 꿈을 꿨어요.
    하늘에 커다란 풍선들이 떠다니고 있었어요.
    그리고 그 풍선을 타고 친구들이랑 모험을 떠났어요.
    """

    model = Model()
    model.updateInfo(text)
    results = model.evaluate_sentiment()

    file_name = "sentiment_analysis_results"
    model.visualize_sentiment(results, file_name)
