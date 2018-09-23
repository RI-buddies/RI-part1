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
    interest_soup = list()
    for tag in divs:
        for div in tag:
            text = div.text
            text = text.replace('\n','')
            if text is not None and not(text in interest_soup):
                if text.find('R$') == -1:
                    interest_soup.append(div)

    for i in interest_soup:
        print(i,'\n---------------------\n')