import os
import time
import pandas as pd
import fasttext
import csv
import re
from preprocess import cusswords_mapping

# 설정
CSV_PATH1 = 'data/train_label0_7319.csv'
CSV_PATH2 = 'data/train_label1_7706.csv'

TRAIN_TXT_PATH = 'chat_fasttext_train_combined.txt'

MODEL_SAVE_DIR = 'models'
TOTAL_EPOCHS = 10
LR = 1
WORD_NGRAMS = 2
DIM = 100


# 로그 함수
def log(msg):
    now = time.strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{now} {msg}")


#  전처리 함수 (학습용)
def prepare_combined_training_data(csv_paths, output_txt_path):
    combined_df = pd.concat([pd.read_csv(p, encoding='utf-8-sig') for p in csv_paths], ignore_index=True)
    combined_df['text'] = combined_df['text'].astype(str).str.strip()
    combined_df['text'] = combined_df['text'].str.replace('"', '', regex=False)
    combined_df['text'] = combined_df['text'].str.replace('\n', ' ', regex=False)
    combined_df['label'] = '__label__' + combined_df['label'].astype(str)

    combined_df['text'] = combined_df['text'].apply(cusswords_mapping)
    
    combined_df[['label', 'text']].to_csv(
        output_txt_path,
        sep=' ',
        index=False,
        header=False,
        quoting=csv.QUOTE_NONE,
        escapechar='\\',
        encoding='utf-8'
    )


# 메인 실행
if __name__ == "__main__":
    prepare_combined_training_data([CSV_PATH1, CSV_PATH2], TRAIN_TXT_PATH)
    log("✅ 두 CSV 합쳐 학습용 txt 생성 완료")

    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    log(f"🚀 FastText 전체 {TOTAL_EPOCHS} 에폭 학습 시작")
    start_time = time.time()

    model = fasttext.train_supervised(
        input=TRAIN_TXT_PATH,
        epoch=TOTAL_EPOCHS,
        lr=LR,
        wordNgrams=WORD_NGRAMS,
        dim=DIM,
        verbose=2, 
        minn=3,
        maxn=6
    )

    model_path = os.path.join(MODEL_SAVE_DIR, f'model_epoch_{TOTAL_EPOCHS}_23th.ft')
    model.save_model(model_path)

    end_time = time.time()
    log(f"\n💾 모델 저장 완료 → {model_path}")
    log(f"⏱️ 전체 학습 시간: {end_time - start_time:.2f}초")
    log("🎉 모델 학습 및 저장 완료")
