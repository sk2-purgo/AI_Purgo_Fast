import fasttext
import os
import re
import time
import pandas as pd
import tracemalloc
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix


# 초성, 변형 욕설 리스트
cusswords_pattern = [
    (r'ㅅ[\W\d_]*ㅂ', '시발'),
    (r'시[\W\d_]*발', '시발'),
    (r'ㅅ[\W\d_]*발', '시발'),
    (r'ㅅ[\W\d_]*벌', '시발'),
    (r'ㅆ[\W\d_]*ㅂ', '씨발'),
    (r'씨[\W\d_]*발', '씨발'),
    (r'ㅆ[\W\d_]*발', '씨발'),
    (r'ㅆ[\W\d_]*벌', '씨발'),

    (r'\bㅁ[\W\d_]*ㅊ\b', '미친'),

    (r'ㅂ[\W\d_]*ㅅ', '병신'),
    (r'병[\W\d_]*신', '병신'),
    (r'ㅄ', '병신'),

    (r'ㅈ[\W\d_]*ㄹ', '지랄'),
    (r'ㅈ[\W\d_]*랄', '지랄'),
    (r'지[\W\d_]*랄', '지랄'),


    (r'ㅈ[\W\d_]*같', '좆같'),
    (r'ㅈ[\W\d_]*망', '좆망'),
    (r'좆[\W\d_]*', '좆'),
    (r'ㅈ[\W\d_]*까', '좆까'),

    (r'색[\W\d_]*끼', '새끼'),
    (r'ㄱ[\W\d_]*ㅅ[\W\d_]*ㄲ', '개새끼'),

    (r'존[\W\d_]*나', '존나'), 
    (r'ㅈ[\W\d_]*ㄴ', '존나'), 
    (r'ㅗ\b', '시발'),
    (r'[^가-힣]놈[^가-힣]', '놈'),
    (r'씨[\W\d_]*발[\W\d_]*년', '씨발년'),
    (r'ㅆ[\W\d_]*ㅂ[\W\d_]*ㄴ', '씨발년'),
    (r'좆[\W\d_]*같[\W\d_]*년', '좆같은년'),
]


# 초성, 변형 욕설 매핑 함수
def cusswords_mapping(text):
    for pattern, replacement in cusswords_pattern:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# 메모리 측정 시작
tracemalloc.start()


# 모델 로딩 시간 측정
load_start = time.time()
model_path = os.path.join("models", "model_epoch_10_23th.ft")
fasttext_model = fasttext.load_model(model_path)
load_time = time.time() - load_start


# 텍스트 전처리 함수
def clean_text(text: str) -> str:
    text = cusswords_mapping(text) 
    return re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text)

# FastText 예측 함수
def detect_fasttext(text: str):
    cleaned_text = clean_text(text)
    start_time = time.perf_counter()
    labels, probabilities = fasttext_model.predict(cleaned_text)
    probabilities = np.asarray(probabilities)
    duration = time.perf_counter() - start_time

    if duration < 0.001:
        duration = 0.001

    detected = [label[9:] for label, prob in zip(labels, probabilities) if prob > 0.5]
    return detected, duration, list(zip(labels, probabilities))

# CSV 파일 로딩 및 분석
def analyze_csv_from_local(path: str):
    if not os.path.exists(path):
        print("❌ CSV 파일이 존재하지 않습니다.")
        return

    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        print("❌ 'text' 또는 'label' 컬럼이 존재하지 않습니다.")
        return

    total_infer_time = 0.0
    total_sentences = 0

    y_true = []
    y_pred = []

    with open("log/23차 last_test_2000_filtered_new 테스트 로그.txt", "w", encoding="utf-8") as f:  # 로그 파일 열기
        for idx, row in df.iterrows():
            text = str(row["text"]).strip()
            true_label = str(row.get("label", "")).strip().lower()

            detected, duration, _ = detect_fasttext(text)

            # 예측 결과 중 첫 번째 라벨을 예측값으로 사용 (없으면 빈 문자열)
            pred_label = detected[0].lower() if detected else ""

            y_true.append(true_label)
            y_pred.append(pred_label)

            total_infer_time += duration
            total_sentences += 1

             # 출력 내용
            output = (
                f"문장 {idx+1}: '{text}'\n"
                f"예측된 레이블: {detected}\n"
                f"추론 시간: {round(duration, 4)}초\n"
                f"실제 레이블: {true_label}\n"
                + "-" * 50 + "\n"
            )

            print(output, end="")      
            f.write(output)           

    throughput = round(total_sentences / total_infer_time, 2) if total_infer_time > 0 else 0.0
    avg_infer_time = total_infer_time / total_sentences if total_sentences > 0 else 0.0

    # sklearn으로 성능 지표 계산
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary',pos_label='1', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary',pos_label='1', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary',pos_label='1', zero_division=0)

    current, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_usage_mb = current / (1024 * 1024)

    summary = {
        "총 문장 수": total_sentences,
        "Accuracy (정확도)": f"{accuracy:.4f}",
        "Precision (정밀도)": f"{precision:.4f}",
        "Recall (재현율)": f"{recall:.4f}",
        "F1-score": f"{f1:.4f}",
        "평균 추론 시간": f"{round(avg_infer_time, 4)}초",
        "전체 추론 시간": f"{round(total_infer_time, 4)}초",
        "처리량": f"{throughput} 문장/초",
        "모델 로딩 시간": f"{round(load_time, 4)}초",
        "메모리 사용량": f"{round(memory_usage_mb, 2)}MB"
    }

    print("✅ 분석 요약:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # confusion matrix 추가
    cm = confusion_matrix(y_true, y_pred, labels=['1', '0'])
    df_cm = pd.DataFrame(cm, index=['실제_욕설', '실제_비욕설'], columns=['예측_욕설', '예측_비욕설'])
    
    print("\n📊 Confusion Matrix:")
    print(df_cm)

# 실행
if __name__ == "__main__":
    csv_path = "D:/ㄹㅇ_fasttext/data/last_test_2000_filtered_new.csv" 
    analyze_csv_from_local(csv_path)
