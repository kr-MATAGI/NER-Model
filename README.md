# De-identification-with-NER
  ## 참조
  - [HuggingFace Transformers](https://huggingface.co/docs/transformers/custom_datasets#tok-ner)
  - [ELECTRA+CRF](https://github.com/Hanlard/Electra_CRF_NER)
  - [BERT+CRF](https://github.com/eagle705/pytorch-bert-crf-ner)

  ## Data Example 비교
   \- 아래와 같은 문장과 NE 태깅이 주어졌을 때  
   
   ![image](https://user-images.githubusercontent.com/30927066/158524780-96cf4b5b-23f7-43a9-a83a-c1ccd77af172.png)
  
   ### monologg/koelectra (naver-ner)
   
   | field  | examples |
   | :----: | :------ |
   | tokens | [CLS] 금 ##석 ##객 ##잔 여러분 , 감사 ##드 ##립니다 . [SEP]  |
   | labels | -100  7 -100 -100 -100  0 -100 0  -100 -100  0  -100 
   
  ### my datasets maker
  | field  | examples |
  | :----: | :------ |
  | tokens | [CLS] 금 ##석 ##객 ##잔 여러분 , 감사 ##드 ##립니다 . [SEP]  |
  | labels | -100 7 8 8 8    0  -100 0  -100 -100  0  -100 |
  
  \- 기존 Tag가 "B"만을 가르켜도 tokenize 했을 때 ##로 이어지는 단어가 된다면, I-ORG를 붙여준다"

   
   
