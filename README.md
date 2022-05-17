# De-identification-with-NER
  ## 참조
  - [HuggingFace Transformers](https://huggingface.co/docs/transformers/custom_datasets#tok-ner)
  - [ELECTRA+CRF](https://github.com/Hanlard/Electra_CRF_NER)
  - [BERT+CRF](https://github.com/eagle705/pytorch-bert-crf-ner)
  - [linear-chin CRF in pytorch](https://towardsdatascience.com/implementing-a-linear-chain-conditional-random-field-crf-in-pytorch-16b0b9c4b4ea)
  
 ## WordPiece를 이용해 Dataset의 Label을 만들기 참조
  - [HuggingFace](https://huggingface.co/docs/transformers/custom_datasets#tok-ner)
  - [StackExchange](https://datascience.stackexchange.com/questions/69640/what-should-be-the-labels-for-subword-tokens-in-bert-for-ner-task)
  - [git/huggingface/issues](https://github.com/huggingface/transformers/issues/323)
  - [discuss Huggingface](https://discuss.huggingface.co/t/converting-word-level-labels-to-wordpiece-level-for-token-classification/2118/6)   
   

## baseline 기록
  - 데이터 : 모두의 말뭉치 2019 (신문/구어 도메인)
  - max_len : 128
  - f1 score
  1. KLUE/BERT-base : 0.9137281805015626
  2. KLUE/BERT-base-LSTM-CRF : 0.9190899917438767
  3. KLUE/RoBERTa-base : 0.9183453182723446
  4. KLUE/RoBERTa-base-LSTM-CRF : ?
  5. monologg/KoCharELECTRA-LAN(Bi_LSTM) : 0.9162957815203087
  6. KLUE/BERT-IDCNN-CRF : 0.9166425426233754
  
 
