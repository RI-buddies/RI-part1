import os
import re
import urllib.request

from bs4 import BeautifulSoup, element

import utils

milson = ("face", {'face': 'Tahoma'})

wallmart = ("div", {'class': 'description-content'})

playtech = ('div', {'id': 'panCaracteristica',
                    'class': 'section about especificacion'})

nova_music = ('div', {
              'class': 'woocommerce-Tabs-panel woocommerce-Tabs-panel--description panel entry-content wc-tab', 'id': 'tab-description'})

mundo_max = ('div', {'id': 'container-features', 'class': 'row'})

casas_bahia = ('div', {'id': 'descricao', 'class': 'descricao'})

multi_som = ('div', {'class': 'boxDescricao'})

made_in_brazil = ('div', {'class': 'paddingbox'})

htmls = [x for x in os.listdir('.') if x.endswith('html')]

for html in htmls:
    print('\n----------------------------------------------------------------------------------------------\n', html, ':')
    soups = utils.soups_of_interest(html)
    sauce = open(html, 'r')
    soup = BeautifulSoup(sauce, 'lxml')
    res = None
    if re.search('Mil sons', html):
        res = soup.findall("face", {'face': 'Tahoma'})
    elif re.search('Wallmart', html):
        res = soup.findall("div", {'class': 'description-content'})
    elif re.search('Playtech', html):
        res = soup.findall(playtech)
    elif re.search('NOVA MUSIC', html):
        div = soup.findall(nova_music)
        res = div.findall('p')[1]
    elif re.search('Mundomax', html):
        res = soup.findall(mundo_max)
    elif re.search('CasasBahia', html):
        div = soup.findall(casas_bahia)
        res = div.findall('p')[1]
    elif re.search('Multisom', html):
        res = soup.findall(multi_som)
    elif re.search('Made in Brazil', html):
        res = soup.findall(made_in_brazil)

    print(res)