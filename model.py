
from transformers import AutoTokenizer, AutoModelForSequenceClassification 
import torch.nn as nn 
from collections import OrderedDict
import torch.nn.functional as F


tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

for param in model.parameters():
    param.requires_grad = False 

# print(model.classifier)

new_classifier = nn.Sequential(OrderedDict([ ('fc1',     nn.Linear(768, 512)),
                                             ('relu1',   nn.ReLU()),
                                             ('fc2',     nn.Linear(512, 512)),
                                             ('relu2',   nn.ReLU()),
                                             ('fc3',     nn.Linear(512, 512)),
                                             ('relu3',   nn.ReLU()),
                                             ('output',  nn.Linear(512, 2)),
                                             ('softmax', nn.LogSoftmax(dim=1))
                                        ]))

model.classifier = new_classifier  

# print(model.classifier)




emb_dim = 300

class Classifier(nn.Module):
    def __init__(self, max_seq_len, emb_dim, hidden1=64):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(max_seq_len*emb_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden1)
        self.fc3 = nn.Linear(hidden1, hidden1)
        self.fc4 = nn.Linear(hidden1, 2)
        self.out = nn.LogSoftmax(dim=1)
    
    def forward(self, inputs):
        x = F.relu(self.fc1(inputs.squeeze(1).float()))
        # x = self.fc1(inputs)


        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return self.out(x)
