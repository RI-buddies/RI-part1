import os
import re
import urllib.request

from bs4 import BeautifulSoup


def soups_of_interest(html):
    sauce = open(html, 'r')
    soup = BeautifulSoup(sauce, 'lxml')
    desc = re.compile('(description|descricao|informations|features|caracteristica|panCaracteristica)')
    divs = list()
    divs.append(soup.find_all(['div', 'section'], {'itemprop': desc}))
    divs.append(soup.find_all(['div', 'section'], {'id': desc}))
    divs.append(soup.find_all(['div', 'section'], {'class': desc}))
    interest_soup = list()
    for tag in divs:
        for div in tag:
            text = div.text
            text = text.replace('\n', '')
            if text is not None and not(div in interest_soup):
                if text.find('R$') == -1:
                    interest_soup.append(div)
    return interest_soup