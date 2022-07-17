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
    <td>BERT+LSTM(POS)+CRF</td><td>????</td>
  </tr>
  <tr>
    <td>ELECTRA+LSTM(POS)+CRF</td><td>93.927</td>
  </tr>
  </table>

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
