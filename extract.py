import os
import re
import urllib.request

from bs4 import BeautifulSoup as BS

htmls = [x for x in os.listdir('.') if x.endswith('html')]
paragraphs = []
for html in htmls:
    print('\n----------------------------------------------------------------------------------------------\n', html,':')
    sauce = open(html, 'r')
    soup = BS(sauce, 'lxml')
    desc = re.compile('(description|descricao|informations|features)')
    divs = list()
    divs.append(soup.find_all(['div','section'], {'itemprop': desc}))
    divs.append(soup.find_all(['div','section'], {'id': desc}))
    divs.append(soup.find_all(['div','section'], {'class': desc}))
    interest_paragraphs = list()
    for tag in divs:
        for div in tag:
            if div.text is not None and not(div.text in interest_paragraphs):
                interest_paragraphs.append(div.text)

    for i in interest_paragraphs:
        print(i,'\n---------------------\n')