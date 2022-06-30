
import pandas as pd 
from torch.utils.data import  DataLoader
from helper_functions import padding, encoder, preprocessing, TrainData, collation_train, collation_test

data = pd.read_csv(r'C:\Users\ritth\code\Strive\Native-Advertising-buid-week\df_extracted.csv')
print(data)


data = data.iloc[:64].reset_index(drop=True)

# print(data.sponsored.value_counts())

train_data = data.iloc[:400].reset_index(drop=True)
test_data  = data.iloc[400:600].reset_index(drop=True)


train_dataset = TrainData(train_data, max_seq_len=256)
test_dataset  = TrainData(test_data,  max_seq_len=256)

# print(train_dataset[0])
# print(train_dataset[1])
# print(train_dataset[2])
# print(train_dataset[3])

train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collation_train)
test_loader  = DataLoader(test_dataset,  batch_size=64, collate_fn=collation_test)


