import json
import os
import re
import urllib.request

from bs4 import BeautifulSoup, element

import utils


def get_text(text):
    text = text.replace('; ', '. ').replace(
        ';', '. ').replace('• ', '. ')
    subs = re.sub(r"([A-Z])", r" \1", text).split()
    new_text = ''
    for i, sub in enumerate(subs):
        new_text = new_text + ' ' + sub
    return new_text


htmls = [x for x in os.listdir('.') if x.endswith('html')]

characteristcs = {'Acabamento': list(), 'Fundo': list(
), 'Cordas': list(), 'Braço': list(), 'Preço': list()}

for html in htmls:
    sauce = open(html, 'r')
    soup = BeautifulSoup(sauce, 'lxml')
    price = ''
    res = None
    if re.search('Mil Sons', html):
        res = soup.findAll('font')
        text = ''
        for r in res:
            text = text + '\n' + r.text
        price = soup.find('b', {'class': 'sale', 'itemprop': 'price'})
        price = price.text.replace('R$', '').replace(' ', '')

    elif re.search('Walmart', html):
        res = soup.findAll('div', {'class': 'description-content'})
        text = res[0].text
        price = soup.find('span', {'class': 'product-price-value'})
        price = price.text.replace('R$', '').replace(' ', '')

    elif re.search('NOVA MUSIC', html):
        div = soup.find('div', {
            'class': 'woocommerce-Tabs-panel woocommerce-Tabs-panel--description panel entry-content wc-tab', 'id': 'tab-description'})
        res = div.findAll('p')[1]
        text = res.text
        price = soup.find('span', {
                          'class': 'woocommerce-Price-amount amount'}).text.replace('R$', '').replace(' ', '')

    elif re.search('Mundomax', html):
        res = soup.find('div', {'id': 'container-features'})
        text = res.text
        price = soup.find('p', {'id': 'info-price'}
                          ).text.replace('R$', '').replace(' ', '')

    elif re.search('CasasBahia', html):
        div = soup.find('div', {'id': 'descricao', 'class': 'descricao'})
        res = div.findAll('p')[1]
        text = res.text
        price = soup.find('i', {'class': 'sale price'}).text.replace(
            'R$', '').replace(' ', '')

    elif re.search('Multisom', html):
        res = soup.find('div', {'id': 'descricao'})
        text = res.text
        price = soup.find('p', {'class': 'prices'})
        price = price.find('ins').text.replace(
            'R$', '').replace(' ', '').replace('Por:', '')

    elif re.search('Made in Brazil', html):
        res = soup.find('div', {'class': 'paddingbox'})
        text = res.text
        price = soup.find('div', {'class': 'precoPor'}).text.replace(
            'R$', '').replace(' ', '')

    else:
        res = soup.find('div', {'id': 'panCaracteristica',
                                'class': 'section about especificacion'})
        text = res.text
        price = soup.find('span', {'id': 'lblPrecoPor', 'class': 'price sale'}).find(
            'strong').text.replace('R$', '').replace(' ', '').replace('Por:', '')

    if type(res) == list():
        text = get_text(text)
    else:
        text = get_text(text)

    characteristcs['Preço'].append((price, html))

    chars = re.findall(
        '[aA-zZ-áàâãéèêíïóôõöúçñ]*:[(| )][(0-9|)]*[aA-zZ-áàâãéèêíïóôõöúçñ]*', text)

    for char in chars:
        if char.split(': ')[0].title() in characteristcs:
            if char.split(': ')[1] != '':
                characteristcs[char.split(': ')[0].title()].append(
                    (char.split(': ')[1].title(), html))

with open('data_specific.json', 'w') as outfile:
    json.dump(characteristcs, outfile)
