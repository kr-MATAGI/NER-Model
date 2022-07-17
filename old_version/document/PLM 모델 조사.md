## KoELECTRA
  
  ### **1. LMKor**  

  - [Git](https://github.com/kiyoungkim1)  
  - [HuggingFace_Electra-base](https://huggingface.co/kykim)  
  - Model Size      
  <img width="839" alt="image" src="https://user-images.githubusercontent.com/30927066/157813763-4b73f00d-e8b3-4f69-968e-6beb2080e80e.png">  
  
  - Data  
    \- 국내 주요 커머스 리뷰 1억개 + ㅂㄹ로그 형 웹사이트 2000만개 (75GB)  
    \- 모두의 말뭉치 (18G)  
    \- 위키피디아와 나무위키 (6GB)  
    \- 불필요하거나 너무 짧은 문장, 중복되는 문장을 제외해 100GB의 데이터 중 70GB (약 127억개의 token)의 텍스트 데이터 사용  
    \- 데이터는 화장품(8GB), 식품(6GB), 전자제품(13GB), 반려동물(2GB) 등의 카테고리로 분류되어 있으며 도메인 특화 언어모델 학습에 사용
  
  - Vocab  
  <img width="808" alt="image" src="https://user-images.githubusercontent.com/30927066/157813874-ce040878-58f0-442f-b43c-6d8e90b09c0e.png">  
  
  - 성능  
  <img width="844" alt="image" src="https://user-images.githubusercontent.com/30927066/157814125-c179516c-7aa4-4dbb-80d9-676de2f7166d.png">

    
  ### **2. 박장원님(monologg) - 현재 사용중인 모델 (v3)**

  - [Git](https://github.com/monologg/KoELECTRA)  
  - [HuggingFace_KoELECTRA](https://huggingface.co/monologg)  
  - Model Size
  <img width="729" alt="image" src="https://user-images.githubusercontent.com/30927066/157815012-577599e6-faab-488c-b0f5-25a30c9e67e5.png">  

  - Data  
    \- v1, v2의 경우 약 14G Corpus (2.6B tokens)를 사용. (뉴스, 위키, 나무위키)  
    \- v3의 경우 약 20G의 모두의 말뭉치를 추가적으로 사용. (신문, 문어, 구어, 메신저, 웹)

  - Vocab  
  <img width="814" alt="image" src="https://user-images.githubusercontent.com/30927066/157815081-5cccca53-cb8b-41e8-93c6-4e161b65bc04.png">  

  - Pretraining Details  
  <img width="840" alt="image" src="https://user-images.githubusercontent.com/30927066/157815190-d3a3187f-4fd5-4135-a469-47789a023a31.png">  

  - 성능
  <img width="845" alt="image" src="https://user-images.githubusercontent.com/30927066/157815244-dca110f7-6317-47a7-a601-000c0f6e12a4.png">
