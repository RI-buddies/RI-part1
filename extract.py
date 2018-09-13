import os
import re
import urllib.request

from bs4 import BeautifulSoup as BS

htmls = [x for x in os.listdir('.') if x.endswith('html')]
paragraphs = []
for html in htmls:
    print(html)
    sauce = open(html, 'r')
    soup = BS(sauce, 'lxml')

    interest_paragraph = None
    for p in soup.find_all('p'):
        if(re.search('Braço', p.text) or re.search('Corpo', p.text) or re.search('Tarraxas', p.text)):
            interest_paragraph = p.text
        
    print(interest_paragraph)
    paragraphs.append(interest_paragraph)
