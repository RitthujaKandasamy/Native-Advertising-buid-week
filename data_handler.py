
import pandas as pd 
from torch.utils.data import  DataLoader
from helper_functions import padding, encoder, preprocessing, TrainData, collation_train, collation_test

data = pd.read_csv('df_extracted_2.csv')

idx = int(0.8 * len(data))

train_data = data.iloc[:idx].reset_index(drop=True)
test_data  = data.iloc[idx:].reset_index(drop=True)


train_dataset = TrainData(train_data, max_seq_len=256)
test_dataset  = TrainData(test_data,  max_seq_len=256)


train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collation_train)
test_loader  = DataLoader(test_dataset,  batch_size=64, collate_fn=collation_test)


