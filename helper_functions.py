
 
import torch 
import pandas as pd
from collections import  Counter # OrderedDict,
from torch.utils.data import  Dataset, DataLoader
import spacy 
from torchtext.vocab import FastText 


path = "C:/Users/zorve/OneDrive/Desktop/GITHUB/GitHub/chap_3_DL_2nd_Time/14. semantic analysis/data/3000test.csv"

"""
def load_data(path):
     data = pd.read_csv(path, header=None, skiprows=1)
     data.rename({1:'sponsored', 2:'rating1', 3:'rating2'}, axis=1, inplace=True)
     data['content'] = data['rating1'] + ' ' + data['rating2']
     data = data.drop([0, 'rating1', 'rating2'], axis=1)
     data.sponsored = data['sponsored'] - 1
     data = data[['sponsored', 'content']]
     return data 

"""


# data = load_data(path)

data = pd.read_csv('df_extracted.csv')
data = data.iloc[:100].reset_index(drop=True)
# print(data)

nlp = spacy.load("en_core_web_sm") 
fasttext = FastText("simple")

def preprocessing(sentence):
    doc = nlp(sentence)
    tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
    return tokens


def token_encoder(token, vec):
    if token == "<pad>":
        return 1
    else:
        try:
            return vec.stoi[token]
        except:
            return 0

def encoder(tokens, vec):
    return [token_encoder(token, vec) for token in tokens]


def padding(list_of_indexes, max_seq_len, padding_index=1):
    output = list_of_indexes + (max_seq_len - len(list_of_indexes))*[padding_index]
    return output[:max_seq_len]


def front_padding(list_of_indexes, max_seq_len, padding_index=0):
    new_out = (max_seq_len - len(list_of_indexes))*[padding_index] + list_of_indexes
    return new_out[:max_seq_len]


class TrainData(Dataset):
    def __init__(self, df, max_seq_len=32): # df is the input df, max_seq_len is the max lenght allowed to a sentence before cutting or padding
        self.max_seq_len = max_seq_len
        
        counter = Counter()
        train_iter = iter(df.content.values)

        self.vec = FastText("simple")

        self.vec.vectors[1] = -torch.ones(self.vec.vectors[1].shape[0]) # replacing the vector associated with 1 (padded value) to become a vector of -1.
        self.vec.vectors[0] = torch.zeros(self.vec.vectors[0].shape[0]) # replacing the vector associated with 0 (unknown) to become zeros
        self.vectorizer = lambda x: self.vec.vectors[x]
        self.labels = df.sponsored
        sequences = [padding(encoder(preprocessing(sequence), self.vec), max_seq_len) for sequence in df.content.tolist()]
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, i):
        assert len(self.sequences[i]) == self.max_seq_len
        return self.sequences[i], self.labels[i]


train_data = data.iloc[:80].reset_index(drop=True)
test_data  = data.iloc[40:90].reset_index(drop=True)


train_dataset = TrainData(train_data, max_seq_len=64)
test_dataset  = TrainData(test_data,  max_seq_len=64)


def collation_train(batch, vectorizer=train_dataset.vectorizer):
    inputs = torch.stack([torch.stack([vectorizer(token) for token in sentence[0]]) for sentence in batch])
    target = torch.LongTensor([item[1] for item in batch]) # Use long tensor to avoid unwanted rounding
    return inputs, target

def collation_test(batch, vectorizer=test_dataset.vectorizer):
    inputs = torch.stack([torch.stack([vectorizer(token) for token in sentence[0]]) for sentence in batch])
    target = torch.LongTensor([item[1] for item in batch]) # Use long tensor to avoid unwanted rounding
    return inputs, target 

# def collation_train(batch, vectorizer=train_dataset.vectorizer):
#     inputs = torch.stack([torch.stack([vectorizer(token) for token in sentence[0]]) for sentence in batch])
#     target = torch.tensor([item[1] for item in batch]).float()
#     return inputs, target

# def collation_test(batch, vectorizer=test_dataset.vectorizer):
#     inputs = torch.stack([torch.stack([vectorizer(token) for token in sentence[0]]) for sentence in batch])
#     target = torch.tensor([item[1] for item in batch]).float()
#     return inputs, target


train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collation_train)
test_loader  = DataLoader(test_dataset, batch_size=64,  collate_fn=collation_test)



# print(data)

