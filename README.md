# A KoBART-Based Approach to Korean Spelling Correction: Impact on Downstream Korean-to-English Translation
This README.md is a supplementary document for the poster. It will be easier to understand if you read it together with the slides.

You can simply run **"run.sh"** to run whole code. 
But don't forget to download the raw data from AIHub first!

## Data
- Source from: [인터페이스(자판/음성)별 고빈도 오류 교정 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71560)
- Train data from the source is used for our train & val data, and validation data from the source is used for our test data.
- '맞춤법오류_자유게시판.json', '맞춤법오류_질문게시판.json', '맞춤법오류_SNS.json', '자주틀리는맞춤법오류.json' were utilized.
- You can check final data at data/final.
- Train: 15,300(10% for validation) Test: 1,550

## Model
- Finetuned KoBART: erroneous sentence as input, corrected sentence as output. 100% of input has error.

## Inference
<img width="843" alt="스크린샷 2025-06-04 오전 10 45 07" src="https://github.com/user-attachments/assets/31c773ad-2e49-42ca-8f86-da4994b77954" />

- Unlike fintuning step, 50% is erroneous and the other 50% is correct. To check if the model can verify which one to correct.
- Pipeline
  1. Correction: First apply correction step by KoBART, then put its output to the input of Opus-MT.
  2. No Correction: Directly put the input to Opus-MT.

## Result
1. [KoBART](https://huggingface.co/gogamza/kobart-base-v2)
- Average Semantic Similarity:    0.962903614082644
- Corpus BLEU (corr_inf_answer):      0.7797
2. [Opus-MT](https://huggingface.co/Helsinki-NLP/opus-mt-ko-en)
- Corpus BLEU (corr_inf_answer):      0.7308
- Corpus BLEU (not_corr_inf_answer): 0.6661

## Limitations
- In this project, I assume that Opus-MT make perfect translation(to generate pseudo label). However, its translation was not that perfect...
- Data was highly informal, and lots were about Korean SAT. 

## Note
I omitted model and data/raw directory due to their large size. Please refer to directory structure below.

```
src
ㄴ1_data.py # organize data. {'ko_error': erroneous sentence, 'ko_cor': corrected sentence, 'en_cor': translated <<corrected>> sentence by Opus-MT, 'num_errors': number of errors}
ㄴ2_train.py # train Ko-BART
ㄴ3_infer_corr.py # Ko-BART
ㄴ4_infer_trans.py # Opus-MT for correction data, and not correction data
ㄴ5_eval_corr.py # Ko-BART
ㄴ6_eval_trans.py # Opus-MT for correction data, and not correction data
data
ㄴraw
  ㄴtrain
  ㄴtest
ㄴfinal
ㄴinferred
model
ㄴbest
ㄴfinal
```
