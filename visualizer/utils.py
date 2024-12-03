from konlpy.tag import Okt
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

def visualize_wordcloud(text, top_n=80, min_length=2, output_file=None):
    """
    텍스트에서 명사를 추출하고, 워드클라우드를 생성 및 저장
    :param text: 텍스트 데이터
    :param top_n: 상위 단어 개수
    :param min_length: 단어 최소 길이
    :param output_file: 저장할 이미지 경로
    """
    # 한글 폰트 설정
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    if not os.path.exists(font_path):

        plt.rc('font', family='NanumGothic')

    # 명사 추출
    okt = Okt()
    nouns = okt.nouns(text)
    count = Counter(nouns)
    word_count = {tag: counts for tag, counts in count.most_common(top_n) if len(tag) >= min_length}

    # 워드클라우드 생성
    wc = WordCloud(background_color='white', width=800, height=600, font_path=font_path)
    cloud = wc.generate_from_frequencies(word_count)

    # 워드클라우드 출력
    plt.figure(figsize=(8, 8))
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    # 이미지 저장
    if output_file:
        cloud.to_file(output_file)
        print(f"워드클라우드 이미지가 저장되었습니다: {output_file}")

def visualize_sentiment_analysis(dates, scores, output_file=None):
    """
    날짜별 감성 점수를 시각화하고 저장
    :param dates: 날짜 리스트
    :param scores: 감성 점수 리스트
    :param output_file: 저장할 이미지 경로
    """
    plt.figure(figsize=(10, 6))
    plt.plot(dates, scores, marker='o', linestyle='-', color='blue', label='Sentiment Score')
    plt.title('Sentiment Scores Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sentiment Score', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # 시각화 저장
    if output_file:
        plt.savefig(output_file)
        print(f"감성 점수 시각화가 저장되었습니다: {output_file}")

    plt.show()

if __name__ == "__main__":
    # 데이터 예시
    name = "lee"
    age = 27
    message = """
    오늘은 태권도장에 가기 싫었어요. 몸도 피곤하고 그냥 집에서 쉬고 싶었어요.
    태권도 수업을 잘 따라가고 싶은데 요즘 마음이 자꾸 무거워요. 엄마한테 가기 싫다고
    솔직하게 말하고 싶었지만 엄마가 속상해하실 것 같아서 그냥 말 못 했어요. 엄마는
    태권도가 저한테 좋은 운동이라고 생각하시고, 관장님도 제게 친절하시지만, 왜 이렇게
    가기 싫은지 저도 잘 모르겠어요. 다음에 가기 싫은 이유를 엄마에게 솔직하게 말할 수
    있으면 좋겠어요.
    """
    output_wordcloud_file = f"SentimentAnalysis_{name}_{age}_wordcloud.png"
    output_chart_file = f"SentimentAnalysis_{name}_{age}_chart.png"

    # 날짜별 감성 점수 예시 데이터
    dates = ['2024-11-20', '2024-11-21', '2024-11-22', '2024-11-23']
    scores = [3, -2, 5, 0]

    # 워드클라우드 생성 및 저장
    generate_wordcloud(message, output_file=output_wordcloud_file)

    # 날짜별 감성 점수 시각화 및 저장
    visualize_sentiment_analysis(dates, scores, output_file=output_chart_file)
