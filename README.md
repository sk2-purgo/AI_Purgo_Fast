# AI_Purgo_Fast

욕설 및 비속어 탐지를 위한 FastText + GPT-3.5 기반 정제 시스템입니다. 욕설 단어를 탐지하고, 정중한 표현으로 순화합니다.

 
---

## ✅ 핵심 기능 요약

| 기능 | 설명 |
|------|------|
| **빠른 욕설 탐지** | FastText를 기반으로 문장 속 `욕설` 여부를 빠르게 탐지합니다. (`__label__1` → 욕설) |
| **변형 욕설 탐지** | `ㅅㅂ`, `ㅈ@ㄹ`, `ㅄ` 등 변형된 욕설도 탐지가 가능합니다. |
| **GPT-3.5 문장 정제** | 욕설이 포함된 문장에 한해 GPT-3.5를 이용하여 **문맥을 유지하며 예의 바른 표현으로 순화**합니다. |
| **FastAPI 기반 API 서비스** | 실시간으로 텍스트를 분석할 수 있는 REST API 제공 (`/analyze` 엔드포인트) |

---

## 🔧 사용 기술 및 라이브러리

| 항목 | 설명 |
|------|------|
| **Python 3.10.10** | 본 프로젝트의 기본 개발 환경입니다. |
| **FastText** | Facebook이 개발한 경량 텍스트 분류 모델. 빠른 학습 및 예측이 가능하며, 한글 데이터에 최적화하기 위해 커스터마이징해 사용합니다. |
| **OpenAI GPT-3.5** | 욕설 문장을 문맥에 맞게 정중하게 순화하기 위해 사용되는 LLM입니다. |
| **FastAPI** | 비동기 처리 기반의 고성능 Python 웹 프레임워크. API 서버로서 텍스트 요청을 받아 분석 및 정제 결과를 반환합니다. |
| **Pandas** | 학습 데이터 전처리 및 분석을 위한 데이터프레임 처리 라이브러리입니다. |
| **Scikit-learn** | FastText 모델 성능 평가 시 정확도, 정밀도, F1-score 등을 계산할 때 사용됩니다. |
| **python-dotenv** | `.env` 파일로부터 OpenAI API 키 등의 환경변수를 안전하게 관리합니다. |

---

## 📁 프로젝트 구조
```
.
├── data/
│   ├── train_label0_7319.csv      # 학습용 CSV 파일
│   └── train_label1_7706.csv       # 학습용 CSV 파일
├── fasttext_gpt.py         # 실시간 욕설 탐지 및 정제 처리
├── fasttext_test.py        # FastText 모델 테스트용
├── fasttext_train.py       # 욕설/비욕설 FastText 모델 학습
├── preprocess.py           # 텍스트 전처리 로직
├── requirements.txt        # 필요한 라이브러리 목록
└── README.md

```

---

## ⚙️ 실행 방법

1. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```

2. **FastText 모델 학습**
    ```bash
    python fasttext_train.py
   ```

3. **FastText 모델 테스트**
    ```bash
    python fasttext_test.py
   ```
   
3. **API 서버 실행**
    ```bash
    uvicorn fasttext_gpt:app --host 0.0.0.0 --port 5000 --reload
   ```
