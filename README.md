# NER-Model (한국어 개체명 인식 모델)

## 프로젝트 소개
이 프로젝트는 한국어 텍스트에서 개체명(Named Entity)을 인식하는 딥러닝 모델을 구현한 것입니다. BERT, ELECTRA 등의 사전학습 모델을 기반으로 하며, LSTM과 CRF 레이어를 추가하여 성능을 향상시켰습니다.

## 주요 기능
- 한국어 텍스트의 개체명 인식
- 다양한 모델 구조 지원 (BERT, ELECTRA + LSTM, CRF)
- 형태소 분석 정보를 활용한 성능 개선
- 국립국어원 개체명 분석 말뭉치 데이터셋 지원

## 설치 방법

### 1. 저장소 클론
```bash
git clone https://github.com/kr-MATAGI/NER-Model.git
cd NER-Model
```

### 2. 가상환경 설정 및 의존성 설치
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 데이터 준비
- ner_input_data.7z 압축 파일을 해제
- 사용할 모델의 config 파일에서 train/dev/test npy 경로 설정

## 사용 방법

### 1. 모델 학습 및 평가
```bash
python run_ner.py
```

## 실험 환경
  <table>
  <th>하이퍼 파라미터</th><th>값</th>
  <tr>
    <td>epoch</td><td>10</td>
  </tr>
  <tr>
    <td>형태소 분석 데이터</td><td>국립국어원 형태 분석 말뭉치 (버전 1.0)</td>
  </tr>
  <tr>
    <td>Train Data Size</td><td>260,099 문장</td>
  </tr>
  <tr>
    <td>Dev Data Size</td><td>37,157 문장</td>
  </tr>
  <tr>
    <td>Test Data Size</td><td>74,315 문장</td>
  </tr>
  <tr>
    <td>Train Batch Size</td><td>64</td>
  </tr>
  <tr>
    <td>Dev/Test Batch Size</td><td>128</td>
  </tr>
  <tr>
    <td>Learning Rate</td><td>5e-5</td>
  </tr>
  </table>

## 성능 기록
  ### 데이터: 국립국어원 개체명 분석 말뭉치 2019 (문어 200만 어절 / 구어 100만 어절)
  
  <table>
    <th>모델</th><th>f1-score</th>
  <tr>
    <td>KLUE/BERT-base</td><td>93.154</td>
  </tr>
  <tr>
    <td>KLUE/BERT-base-LSTM(POS)</td><td>93.725</td>
  </tr>
  <tr>
    <td>KLUE/BERT-base-CRF(POS)</td><td>93.825</td>
  </tr>
  <tr>
    <td>KLUE/BERT-base-LSTM(POS)-CRF</td><td>94.019</td>
  </tr>
  <tr>
    <td>monologg/koELECTRA-LSTM(POS)-CRF</td><td>94.264</td>
  </tr>
  
  </table>
  
  ### 데이터: 국립국어원 개체명 분석 말뭉치 2020 (문어 200만 어절 /구어 100만 어절)
  
  <table>
  <th>모델</th><th>f1-score</th>
  <tr>
    <td>BERT+LSTM(POS)+CRF</td><td>93.831</td>
  </tr>
  <tr>
    <td>ELECTRA+LSTM(POS)+CRF</td><td>93.927</td>
  </tr>
  </table>

## <우리말 샘> 사전 라이센스 관련
  - [우리말샘 저작권 정책](https://opendict.korean.go.kr/service/copyrightPolicy)

# 프로젝트 구현시 참고한 것들...
## 모델 구현관련 참고
  - [HuggingFace Transformers](https://huggingface.co/docs/transformers/custom_datasets#tok-ner)
  - [ELECTRA+CRF](https://github.com/Hanlard/Electra_CRF_NER)
  - [BERT+CRF](https://github.com/eagle705/pytorch-bert-crf-ner)
  - [linear-chin CRF in pytorch](https://towardsdatascience.com/implementing-a-linear-chain-conditional-random-field-crf-in-pytorch-16b0b9c4b4ea)

## WordPiece를 이용해 Dataset의 Label을 만들기 참고
  - [HuggingFace](https://huggingface.co/docs/transformers/custom_datasets#tok-ner)
  - [StackExchange](https://datascience.stackexchange.com/questions/69640/what-should-be-the-labels-for-subword-tokens-in-bert-for-ner-task)
  - [git/huggingface/issues](https://github.com/huggingface/transformers/issues/323)
  - [discuss Huggingface](https://discuss.huggingface.co/t/converting-word-level-labels-to-wordpiece-level-for-token-classification/2118/6)   

## Feature Embedding 참고
  - [임베딩을 합치는 4가지 방법](https://jeonghyeokpark.netlify.app/pytorch/2020/12/17/pytorch5.html)

## 프로젝트 구조
```
NER-Model/
├── config/           # 모델 설정 파일
├── data/            # 데이터셋 저장 디렉토리
├── model/           # 모델 구현 파일
├── utils/           # 유틸리티 함수
├── ner_def.py       # 개체명 정의
├── ner_utils.py     # NER 관련 유틸리티
├── run_ner.py       # 메인 실행 파일
└── requirements.txt # 의존성 목록
```

## 개발 환경
- Python 3.7+
- PyTorch 1.10.0
- Transformers 4.16.2
- CUDA 11.3 (GPU 사용시)

## 라이센스
- 이 프로젝트는 MIT 라이센스를 따릅니다.
- 우리말샘 사전 데이터는 [우리말샘 저작권 정책](https://opendict.korean.go.kr/service/copyrightPolicy)을 따릅니다.

## 참고 문헌
