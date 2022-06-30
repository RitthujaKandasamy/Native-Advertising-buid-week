
import os 
import requests 
from bs4 import BeautifulSoup 
import re 
import codecs
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd 



path_csv = "C:/Users/zorve/OneDrive/Desktop/native-advertising/"
data_1 = pd.read_csv(path_csv + "sampleSubmission_v2.csv")
data_2 = pd.read_csv(path_csv + "sampleSubmission.csv")
data_3 = pd.read_csv(path_csv + "train.csv")
data_4 = pd.read_csv(path_csv + "train_v2.csv")



# COMBINIG THEM INTO ONE DATA FRAME

DATA_1 = pd.concat([data_1, data_2, data_3, data_4])


file_name, all_content = [], []


folder_names = "C:/Users/zorve/OneDrive/Desktop/native-advertising/"

# file_name, url_links = [], []

for idx in range(0, 6, 1):
    for file in os.listdir(f'{folder_names}{idx}'):
        path = f'{folder_names}{idx}/' + file 
        # print(path)

        with open(path, 'r') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')

        results = soup.find_all('p')
        reviews = [result.text for result in results]

        reviews = [result.text.replace('\n', ' ').replace('\t', ' ').replace("'", ' ').replace("*", ' ').replace("'", ' ').strip().replace('   ', '').replace('  ', ' ') for result in results]

        # print(reviews)

        one_sentence = ''

        for item in reviews:
            item = item.strip().replace('\n', ' ')
            one_sentence = one_sentence + item + ' '

        file_name.append(file)
        all_content.append(one_sentence)


DATA_2 = pd.DataFrame( { 'file_name' : file_name, 'content' : all_content })

df_merged = pd.merge(DATA_1, DATA_2, left_on='file', right_on='file_name')

processed_data = df_merged[['content', 'sponsored']]

processed_data.to_csv('EXTRACTED_DATA.csv')
print(processed_data)


df_extracted = pd.read_csv('EXTRACTED_DATA.csv')

df_extracted = df_extracted.drop_duplicates()
df_extracted = df_extracted.dropna()
df_extracted.to_csv('df_extracted.csv')
print(df_extracted)
















