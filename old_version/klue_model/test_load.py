import torch
from transformers import AutoModelForTokenClassification, AutoModel

if "__main__" == __name__:
    a = torch.load("./model/tapas_1.bin")
    print(a["bert.embeddings.position_ids"].shape)

    b = AutoModel.from_pretrained("klue/roberta-base")

    print(b.state_dict()["embeddings.position_ids"].shape)