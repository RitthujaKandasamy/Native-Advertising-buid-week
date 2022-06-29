import torch.nn as nn
from transformers import BertModel



bert_model = BertModel.from_pretrained('bert-base-uncased')

class Classifier(nn.Module):
    def __init__(self, bert_model):
        super(Classifier, self).__init__()
        self.emb = bert_model 
        self.fc = nn.Linear(768, 1)
    
    def forward(self, ids, mask, token_type_ids):
        output = self.emb(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output = output.pooler_output
        output = self.fc(output)
        return output

model = Classifier(bert_model)





