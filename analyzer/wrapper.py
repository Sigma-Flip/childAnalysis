import datetime
from model import Model

model = Model()

class LLMWrapper:
    def __init__(self, text: str):
        """
        단일 텍스트 데이터를 기반으로 LLM 작업을 수행하는 클래스.

        Args:
            text (str): 텍스트 데이터.
        """
        self.text = text
        self.contents = self.text.split("\n")  # 줄바꿈 기준으로 리스트로 분리
        self._summary = ""
        self._evaluation = ""
        self._multi_evaluation = ""
        self._analysis = ""
        self._timeline = ""
        self.datetime = datetime.datetime.now()

        model.updateInfo(self.text)  # Model에 텍스트 데이터 전달

    def showMetaInfo(self):
        """
        메타 정보를 출력합니다.
        """
        print("[Meta Information]")
        print(f"Date: {self.datetime.strftime('%Y-%m-%d')}, Time: {self.datetime.strftime('%H:%M:%S')}")

    def showResult(self, result_type):
        """
        결과 유형을 출력합니다.

        Args:
            result_type (str): 결과 유형 (e.g., "Summary", "Evaluation").
        """
        print(f"\n[{result_type}]")

    def summarize(self):
        """
        텍스트 데이터를 요약합니다.
        """
        self._summary = model.create(work='Summary')
        return self._summary

    def showSummary(self):
        self.showResult("Summary")
        if not self._summary:
            self.summarize()
        print(self._summary)

    def evaluate(self):
        """
        텍스트 데이터를 평가합니다.
        """
        self._evaluation = model.create(work="Evaluation")
        return self._evaluation

    def showEvaluation(self):
        self.showResult("Evaluation")
        if not self._evaluation:
            self.evaluate()
        print(self._evaluation)

    def multiEvaluate(self):
        """
        텍스트 데이터를 60가지 감정으로 평가합니다.
        """
        self._multi_evaluation = model.create(work="multiEvaluation")
        return self._multi_evaluation

    def showMultiEvaluation(self):
        self.showResult("Multi-Evaluation")
        if not self._multi_evaluation:
            self.multiEvaluate()
        print(self._multi_evaluation)

    def analyze(self):
        """
        텍스트 데이터를 분석합니다.
        """
        self._analysis = model.create(work='Analysis')
        return self._analysis

    def showAnalysis(self):
        self.showResult("Analysis")
        if not self._analysis:
            self.analyze()
        print(self._analysis)

    def createTimeline(self):
        """
        타임라인을 생성합니다.
        """
        self._timeline = model.create(work='Timeline')
        return self._timeline

    def showTimeline(self):
        self.showResult("Timeline")
        if not self._timeline:
            self.createTimeline()
        print(self._timeline)

    def showAll(self):
        """
        모든 결과를 출력합니다.
        """
        print("[Full Report]")
        self.showMetaInfo()
        self.showSummary()
        self.showEvaluation()
        self.showMultiEvaluation()
        self.showAnalysis()
        self.showTimeline()
        print("[End of Report]")

# 실행 예시
if __name__ == "__main__":
    text = """
오늘은 정말 재미있는 꿈을 꿨어요.
하늘에 커다란 풍선들이 떠다니고 있었어요.
그리고 그 풍선을 타고 친구들이랑 모험을 떠났어요.
숲속을 지나서 무지개 다리를 건넜어요.
어디선가 고양이들이 나타나 우리와 함께 놀았어요.
그리고 신기한 나무를 만났어요. 나무가 말을 하더라고요.
나무가 내 이름을 부르는데 정말 놀랐어요!
마지막엔 바다로 갔는데 물고기들이 춤을 추고 있었어요.
그 꿈에서 깼을 때, 너무 아쉬웠어요.
언젠가 진짜로 그런 모험을 떠나고 싶어요.
""".strip()

    parent = LLMWrapper(text=text)
    parent.showMultiEvaluation()
