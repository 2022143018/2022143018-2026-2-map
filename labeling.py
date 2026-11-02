"""
지도 앱 리뷰 전처리 + 라벨링 + 학습 데이터셋 구축

입력 : map_app_reviews_2024_2025_50000.csv  (test1.py로 수집한 원본 50,000건)
출력 : dataset/labeled_reviews.csv          (라벨링된 학습 데이터셋)
       dataset/labeling_summary.csv         (앱별 라벨 분포 요약)
       labeling_results/*.png               (라벨 분포 그래프)

라벨 1. sentiment (감성)  : 평점 기반  4~5점 긍정 / 3점 중립 / 1~2점 부정
라벨 2. topic (주제)      : 키워드 사전 기반, 리뷰가 무엇에 대해 말하는지
"""
import os
import re
import platform

import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 0. 설정
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "map_app_reviews_2024_2025_50000.csv")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
FIG_DIR = os.path.join(BASE_DIR, "labeling_results")
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MIN_LENGTH = 10          # 10글자 미만 리뷰는 주제를 판단할 정보가 부족하여 제외
MIN_KOREAN_RATIO = 0.5   # 글자 중 한글 비율 50% 이상만 사용



if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
else:
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

# ------------------------------------------------------------------
# 1. 라벨 정의
# ------------------------------------------------------------------
SENTIMENT_ORDER = ["긍정", "중립", "부정"]


def label_sentiment(rating):
    if rating >= 4:
        return "긍정"
    if rating == 3:
        return "중립"
    return "부정"


# 주제 키워드 사전 (부정 리뷰 상위 빈출 단어 + 도메인 지식으로 1차 구성,
# 검수 샘플 300건(개발용)의 오분류 사례를 보고 2차 보완)
# - 리뷰 텍스트를 소문자로 바꾼 뒤 키워드가 몇 번 나오는지 센다.
# - STRONG_KEYWORDS 는 주제가 분명한 단어라서 3점, 나머지는 1점.
# - 점수가 가장 높은 주제를 대표 주제(topic)로, 1점 이상인 주제를 모두 topic_all 에 기록한다.
# - 점수가 같으면 아래 사전 순서(앞쪽 우선)를 따르고, 아무 키워드도 없으면 '일반평가'.
TOPIC_KEYWORDS = {
    "기업이슈": ["명복", "고인", "괴롭힘", "불매", "갑질", "여성혐오", "웹툰", "보이콧",
              "죽여", "죽음", "죽인", "사람죽"],
    "대중교통": ["버스", "지하철", "정류장", "정류소", "대중교통", "호선", "환승", "도착정보",
              "도착 정보", "배차", "막차", "첫차", "열차", "전철", "노선",
              "승하차", "하차", "승차", "도착알림", "도착 알림"],
    "길안내": ["네비", "내비", "길찾기", "길 찾기", "경로", "길안내", "길 안내", "길을", "길로",
            "목적지", "고속도로", "자전거", "도보", "운전", "차선", "우회전", "좌회전", "유턴",
            "과속", "단속", "카메라", "음성안내", "음성 안내", "재탐색", "출발지", "빠른길",
            "느린길", "최단", "막히", "톨게이트", "주차", "이륜차", "오토바이", "어린이보호구역",
            "안드로이드오토", "안드로이드 오토", "입구", "교통정보", "휴게소"],
    "위치GPS": ["위치", "gps", "지피에스", "나침반", "나침판", "방향", "터널", "엉뚱한 곳"],
    "장소정보": ["검색", "장소", "가게", "주소", "즐겨찾기", "영업", "로드뷰", "로드맵", "거리뷰",
              "리뷰", "맛집", "상호", "매장", "폐업", "건물", "식당", "카페", "별점", "충전소",
              "지도 업데이트", "지도업데이트", "지도 정보", "폴더"],
    "오류성능": ["오류", "에러", "튕", "꺼짐", "꺼져", "꺼지", "멈춤", "멈춰", "먹통", "느려",
              "느림", "로딩", "버벅", "렉", "네트워크", "접속", "실행이", "실행 안", "강제종료",
              "배터리", "발열", "안켜", "안 켜", "끊겨", "끊기", "끊김", "깨져", "서버", "장애",
              "중지", "초기화면", "버그"],
    "업데이트UI": ["업데이트", "업뎃", "화면", "버튼", "디자인", "메뉴", "글씨", "글자", "다크모드",
                "위젯", "인터페이스", "ui", "개편", "바뀌", "바꿔", "바꼈", "아이콘", "예전",
                "이전 버전", "원상복구", "롤백", "다운그레이드"],
    "광고정책": ["광고", "배너", "로그인", "로그아웃", "개인정보", "권한", "약관", "동의",
              "회원가입", "팝업", "강제 설치", "결제"],
}
STRONG_KEYWORDS = {
    "기업이슈": ["명복", "고인", "괴롭힘", "불매", "여성혐오", "죽여", "사람죽"],
    "대중교통": ["대중교통", "승하차", "하차알림", "하차 알림", "도착알림"],
    "위치GPS": ["gps", "지피에스", "터널"],
    "장소정보": ["지도 업데이트", "지도업데이트"],
    "광고정책": ["광고", "개인정보"],
}
TOPIC_ORDER = list(TOPIC_KEYWORDS.keys()) + ["일반평가"]


def label_topic(text):
    text = text.lower()
    scores = {}
    for topic, words in TOPIC_KEYWORDS.items():
        score = sum(text.count(w) for w in words)
        score += 2 * sum(text.count(w) for w in STRONG_KEYWORDS.get(topic, []))
        scores[topic] = score
    best = max(scores.values())
    if best == 0:
        return "일반평가", ""
    hits = [t for t in TOPIC_KEYWORDS if scores[t] > 0]
    primary = next(t for t in TOPIC_KEYWORDS if scores[t] == best)
    return primary, "|".join(hits)


# ------------------------------------------------------------------
# 2. 전처리
# ------------------------------------------------------------------
df = pd.read_csv(INPUT_FILE)
n_raw = len(df)

df["review_text"] = df["review_text"].fillna("").astype(str).str.strip()
df["review_text"] = df["review_text"].str.replace(r"\s+", " ", regex=True)

korean = df["review_text"].str.count(r"[가-힣ㄱ-ㅎㅏ-ㅣ]")
letters = df["review_text"].str.count(r"[^\W\d_]")
korean_ratio = korean / letters.where(letters > 0, 1)

steps = [("원본", n_raw)]

df = df.drop_duplicates(subset=["review_id"])
steps.append(("review_id 중복 제거", len(df)))

mask_korean = (letters.loc[df.index] > 0) & (korean_ratio.loc[df.index] >= MIN_KOREAN_RATIO)
df = df[mask_korean]
steps.append(("한글 위주 리뷰만 (한글 50% 이상)", len(df)))

df = df[df["review_text"].str.len() >= MIN_LENGTH].copy()
steps.append((f"{MIN_LENGTH}글자 이상", len(df)))

print("=" * 60)
print("전처리 단계별 데이터 수")
print("=" * 60)
for name, n in steps:
    print(f"{name:<28} {n:>8,}건")

# ------------------------------------------------------------------
# 3. 라벨링
# ------------------------------------------------------------------
df["sentiment"] = df["rating"].apply(label_sentiment)
topics = df["review_text"].apply(label_topic)
df["topic"] = [t[0] for t in topics]
df["topic_all"] = [t[1] for t in topics]
df["review_length"] = df["review_text"].str.len()

cols = ["app_name", "review_id", "review_date", "rating", "review_text",
        "review_length", "thumbs_up_count", "app_version",
        "sentiment", "topic", "topic_all"]
df = df[cols].sort_values(["app_name", "review_date"], ascending=[True, False])
df.to_csv(os.path.join(DATASET_DIR, "labeled_reviews.csv"),
          index=False, encoding="utf-8-sig")

# 검수용 샘플은 label_check.py 에서 만들고 평가한다.

# ------------------------------------------------------------------
# 4. 라벨 분포 요약
# ------------------------------------------------------------------
sent_tab = pd.crosstab(df["app_name"], df["sentiment"])[SENTIMENT_ORDER]
topic_tab = pd.crosstab(df["topic"], df["app_name"]).reindex(TOPIC_ORDER)
neg = df[df["sentiment"] == "부정"]
neg_topic_ratio = (pd.crosstab(neg["topic"], neg["app_name"], normalize="columns") * 100).reindex(TOPIC_ORDER)

summary = pd.concat({"감성(건)": sent_tab.T, "주제(건)": topic_tab}, axis=0)
summary.to_csv(os.path.join(DATASET_DIR, "labeling_summary.csv"), encoding="utf-8-sig")

print("\n최종 학습 데이터셋:", f"{len(df):,}건")
print(df["app_name"].value_counts().to_string())
print("\n[감성 라벨]")
print(sent_tab)
print("\n[주제 라벨]")
print(topic_tab)
print("\n[부정 리뷰의 주제 비율(%)]")
print(neg_topic_ratio.round(1))

# ------------------------------------------------------------------
# 5. 그래프
# ------------------------------------------------------------------
ax = (pd.crosstab(df["app_name"], df["sentiment"], normalize="index")[SENTIMENT_ORDER] * 100).plot(
    kind="bar", stacked=True, figsize=(8, 5), color=["#4C9F70", "#B0B0B0", "#D9534F"])
ax.set_title("앱별 감성 라벨 비율")
ax.set_xlabel("앱")
ax.set_ylabel("비율 (%)")
plt.xticks(rotation=0)
plt.legend(title="감성", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "L1_앱별_감성_비율.png"), dpi=200)
plt.close()

ax = topic_tab.plot(kind="barh", figsize=(9, 6))
ax.set_title("앱별 주제 라벨 개수")
ax.set_xlabel("리뷰 개수")
ax.set_ylabel("주제")
ax.invert_yaxis()
ax.legend(title="앱")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "L2_앱별_주제_개수.png"), dpi=200)
plt.close()

ax = neg_topic_ratio.drop("일반평가").plot(kind="barh", figsize=(9, 6))
ax.set_title("부정 리뷰(1~2점)의 주제 비율 (일반평가 제외)")
ax.set_xlabel("앱별 부정 리뷰 중 비율 (%)")
ax.set_ylabel("주제")
ax.invert_yaxis()
ax.legend(title="앱")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "L3_부정리뷰_주제_비율.png"), dpi=200)
plt.close()

topic_rating = df.pivot_table(index="topic", columns="app_name", values="rating", aggfunc="mean").reindex(TOPIC_ORDER)
ax = topic_rating.plot(kind="barh", figsize=(9, 6))
ax.set_title("주제별 평균 평점")
ax.set_xlabel("평균 평점")
ax.set_ylabel("주제")
ax.set_xlim(1, 5)
ax.invert_yaxis()
ax.legend(title="앱")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "L4_주제별_평균_평점.png"), dpi=200)
plt.close()

print("\n저장 완료:", DATASET_DIR, FIG_DIR)
