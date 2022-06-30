

import requests 
from bs4 import BeautifulSoup 
import re 
import codecs
from urllib.request import urlopen
from bs4 import BeautifulSoup


with open("C:/Users/zorve/OneDrive/Desktop/native-advertising/0/10146_raw_html.txt", 'r') as f:
    content = f.read()

soup = BeautifulSoup(content, 'html.parser')

results = soup.find_all('p')
reviews = [result.text for result in results]

print(len(reviews))



























# request = requests.get('https://www.yelp.com/biz/social-brew-cafe-pyrmont')
# soup = BeautifulSoup(request.text, 'html.parser')

# regex = re.compile('.*comment.*')
# results = soup.find_all('p', {'class':regex})
# reviews = [result.text for result in results]

# print(reviews)