from fastapi import FastAPI
from pydantic import BaseModel  
from fastapi.responses import JSONResponse  
import fasttext  
import os  
import re   
from openai import OpenAI  
from dotenv import load_dotenv 


# 환경변수 로드 (.env 파일에서 OPENAI_API_KEY 읽어오기)
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # OpenAI 클라이언트 초기화


# FastText 모델 로딩
model_path = os.path.join("models", "model_epoch_10_23th.ft")  
fasttext_model = fasttext.load_model(model_path)  


# 초성욕 강제 정규화 딕셔너리
초성욕_대응표 = {
    "ㅅㅂ": "시발",
    "ㅆㅂ": "씨발",
    "ㅁㅊ": "미친",
    "ㅂㅅ": "병신",
    "ㅄ": "병신",
    "ㅈㄹ": "지랄",
    "ㄱㅅㄲ": "개새끼",
    "ㅈㄴ": "존나",
    "ㅈ같": "좆같",
    "ㅈ까": "좆까",
}


# 텍스트 정규화: 초성욕 변환 + 불필요 문자 제거
def normalize_text(text: str) -> str:
    text = text.strip()

    # 초성욕 정규화
    for 초성, 치환어 in 초성욕_대응표.items():
        text = re.sub(rf'\b{re.escape(초성)}\b', 치환어, text)

    # 불필요 문자 제거 (한글, 숫자, 영문, 공백만 남김)
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text)

    return text


# FastText 욕설 감지 함수 (오직 __label__1만 bad로 간주)
def detect_fasttext(text: str):
    cleaned_text = normalize_text(text)  # 전처리 수행
    labels, probabilities = fasttext_model.predict(cleaned_text)  # 예측 수행
    # __label__1이고 확률 > 0.4인 경우에만 bad로 감지
    detected = [label.replace("__label__", "")
                for label, prob in zip(labels, probabilities)
                if label == "__label__1" and prob > 0.4]
    return detected


# GPT-3.5를 이용한 욕설 순화(정제) 함수
def rewrite_text_gpt3_5(text: str) -> str:
    prompt = f"""
당신은 사람들의 발화를 정중하고 긍정적인 표현으로 순화하는 AI입니다.

다음 문장을 아래 조건에 맞춰 순화해주세요:

[💬 순화 조건]
1. 문장에 욕설, 비속어, 혐오, 성적인 표현이 있을 경우 → 문맥을 고려하여 **정중하고 바른 표현**으로 자연스럽게 바꾸세요.
2. 노골적이지 않아도 공격적이거나 부정적인 뉘앙스를 가진 단어는 **긍정적이고 포용적인 표현**으로 순화하세요.
3. **문장의 구조와 말투는 유지**하면서, 문제가 되는 단어만 바꾸는 것이 핵심입니다.
4. 예의 바르고 부드러운 어투로 작성하세요.
5. **욕설이 없는 경우에는 문장을 수정하지 않고 그대로 반환**하세요.
6. 출력은 반드시 **정제된 문장 한 줄만**, 설명이나 따옴표 없이 출력하세요.

[🔴 반드시 순화해야 할 표현 예시]
- 초성 욕설 (예: ㅅㅂ, ㅈㄹ, ㅄ 등)
- 감정 과격 표현 (예: 존나, 개같은, 지랄 등)
- 인신 공격 표현 (예: 미친놈, 병신, 새끼, 븅신 등)

[🧠 순화 예시]
- "씨발 오늘 왜 이래" → "아 진짜 오늘 왜 이래"
- "개같은 새끼" → "정말 못된 사람"
- "존나 짜증나" → "정말 짜증나"
- "지랄하지 마" → "화내지 마"
- "병신같이 하네" → "어수룩하게 하네"
- "ㅅㅂ" → "아 진짜"
- "저새끼 뭐야" → "저 사람 뭐지"

문장: "{text}"
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "너는 문장에서 비속어, 욕설, 혐오 표현을 찾아 정중한 표현으로 순화하는 편집기야. 문장의 말투와 구조는 그대로 유지하고, 설명 없이 결과 문장만 출력해."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
        )
        result = response.choices[0].message.content.strip()

        # 양쪽 따옴표가 있을 경우 제거
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]
        return result
    except Exception as e:
        print(f"❌ GPT 정제 실패: {e}")
        return text


# FastAPI 앱 초기화
app = FastAPI()


# 요청 바디 검증 모델 정의
class TextRequest(BaseModel):
    text: str  


# 기본 루트 엔드포인트
@app.get("/")
async def root():
    return {"message": "✅ FastText + GPT-3.5 욕설 탐지 서버 실행 중!"}


# 분석용 POST 엔드포인트
@app.post("/analyze")
async def analyze(request: TextRequest):
    text = request.text.strip()  
    print(f"입력 문장: {text}")

    fasttext_result = detect_fasttext(text)  
    fasttext_hit = 1 if fasttext_result else 0  
    print(f"🔍 FastText 탐지 결과: {fasttext_result}")

    # 기본 응답 구조
    response = {
        "fasttext": {"is_bad": fasttext_hit, "detected_words": fasttext_result},
        "result": {"original_text": text, "rewritten_text": text},
        "final_decision": 0
    }

    # FastText가 bad로 판단한 경우에만 GPT 정제
    if fasttext_hit:
        print("✅ FastText 욕설 감지 → GPT 정제 시작")
        rewritten = rewrite_text_gpt3_5(text)
        response["result"]["rewritten_text"] = rewritten
        response["final_decision"] = 1
    else:
        print("⭕ 확률 기준 미달로 GPT 정제 없이 원문 반환")

    return JSONResponse(content=response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fasttext_gpt:app", host="0.0.0.0", port=5000, reload=True)