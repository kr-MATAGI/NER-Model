## Config

  \- model name: monologg/koelectra-small-v3-discriminator  
  \- architectures: ElectraCRF_NER  
  \- attention probs dropout prob: 0.1  
  \- embedding size: 128  
  \- hidden act: gelu  
  \- hidden dropout prob: 0.1  
  \- hidden size: 256  
  \- max position embeddings: 512  
  \- num attention heads: 4  
  \- num hidden layers: 12  
  \- vocab size: 35000  
  
  \- train epoch: 20    
  \- learning rate: 5e-5  
  \- optimizer: AdamW  
  \- train batch size: 32  
  \- eval batch size: 128  
  
## Datasets size
  \- Train : 81,000  
  \- Dev : 9,000  
  
## Results
  
  - epoch 20  

  - f1 = 0.8405462649764486
  - loss = 5.255626984045539
  - precision = 0.811679589718542
  - recall = 0.8715418906516323
  
|           | precision | recall  | f1-score  | support |
| :-------: | :-------: | :-----: | :-------: | :-----: |
| AFW        |  0.56     | 0.62    | 0.59      | 573     |
| ANM        |  0.70     | 0.78    | 0.74      | 636     |
| CVL        |  0.79     | 0.85    | 0.82      | 5518    |
| DAT        |  0.91     | 0.96    | 0.94      | 3303    |
| EVT        |  0.82     | 0.87    | 0.84      | 1638    |
| FLD        |  0.49     | 0.63    | 0.55      | 230     |
| LOC        |  0.78     | 0.83    | 0.81      | 2048    |
| MAT        |  0.38     | 0.13    | 0.19      | 23      |
| NUM        |  0.91     | 0.95    | 0.93      | 5748    |
| ORG        |  0.85     | 0.84    | 0.85      | 4182    |
| PER        |  0.85     | 0.88    | 0.87      | 4197    |
| PLT        |  0.53     | 0.47    | 0.50      | 17      |
| TIM        |  0.87     | 0.95    | 0.91      | 418     |
| TRM        |  0.68     | 0.77    | 0.72      | 1977    |
| -         |  0.00     | 0.00    | 0.00      | 0      |
|           |           |         |           |         |
| micro avg | 0.81      | 0.87    | 0.84      | 30508   |
| macro avg | 0.68      | 0.70    | 0.68      | 30508   |
| weighted avg | 0.83   | 0.87    | 0.85      | 30508   |

