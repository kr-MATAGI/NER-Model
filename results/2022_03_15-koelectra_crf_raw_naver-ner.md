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

  - f1 = 0.8422956797015221
  - loss = 5.166016205935411
  - precision = 0.8028882422674857
  - recall = 0.8857711856849865
  
|            | precision | recall  | f1-score  | support |
| :--------: | :-------: | :-----: | :-------: | :-----: |
| AFW        |  0.53     | 0.64    | 0.58      | 624     |
| ANM        |  0.63     | 0.79    | 0.70      | 649     |
| CVL        |  0.76     | 0.87    | 0.81      | 5940    |
| DAT        |  0.91     | 0.96    | 0.93      | 3450    |
| EVT        |  0.79     | 0.88    | 0.83      | 1859    |
| FLD        |  0.44     | 0.68    | 0.53      | 233     |
| LOC        |  0.78     | 0.85    | 0.81      | 2190    |
| MAT        |  0.27     | 0.25    | 0.26      | 24      |
| NUM        |  0.89     | 0.95    | 0.92      | 6411    |
| ORG        |  0.84     | 0.87    | 0.86      | 4726    |
| PER        |  0.86     | 0.91    | 0.88      | 4803    |
| PLT        |  0.21     | 0.47    | 0.29      | 17      |
| TIM        |  0.87     | 0.97    | 0.92      | 432     |
| TRM        |  0.60     | 0.76    | 0.67      | 2285    |
| -          |  0.00     | 0.00    | 0.00      | 0      |
|            |           |         |           |         |
| micro avg  | 0.80      | 0.89    | 0.84      | 33643   |
| macro avg  | 0.62      | 0.72    | 0.67      | 33643   |
| weighted avg | 0.81   | 0.89    | 0.84      | 33643   |
