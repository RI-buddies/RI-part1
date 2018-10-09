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


def do_specific(htmls):
    characteristcs = {'Acabamento': dict(), 'Fundo': dict(
    ), 'Cordas': dict(), 'Braço': dict(), 'Preço': dict()}

    for html in htmls:
        print(html,'-----------\n')
        sauce = open(html, 'r')
        soup = BeautifulSoup(sauce, 'lxml')
        price = ''
        res = None
        if re.search('(Mil Sons|A)', html.split('/')[-1]):
            res = soup.findAll('font')
            text = ''
            for r in res:
                text = text + '\n' + r.text
            try:
                price = soup.find('b', {'class': 'sale', 'itemprop': 'price'})
                price = price.text.replace('R$', '').replace(' ', '')
            except AttributeError:
                price = None            

        elif re.search('(Walmart|G)', html.split('/')[-1]):
            res = soup.findAll('div', {'class': 'description-content'})
            text = res[0].text
            try:
                price = soup.find('span', {'class': 'product-price-value'})
                price = price.text.replace('R$', '').replace(' ', '')
            except AttributeError:
                price = None

        elif re.search('(NOVA MUSIC|B)', html.split('/')[-1]):
            div = soup.find('div', {
                'class': 'woocommerce-Tabs-panel woocommerce-Tabs-panel--description panel entry-content wc-tab', 'id': 'tab-description'})
            res = div.findAll('p')[1]
            text = res.text
            try:
                price = soup.find('span', {
                    'class': 'woocommerce-Price-amount amount'}).text.replace('R$', '').replace(' ', '')
            except AttributeError:
                price = None

        elif re.search('(Mundomax|C)', html.split('/')[-1]):
            res = soup.find('div', {'id': re.compile('(container-features|infoProd)')})
            text = res.text
            try:
                price = soup.find('p', {'id': 'info-price'}
                              ).text.replace('R$', '').replace(' ', '')
            except AttributeError:
                price = None

        elif re.search('(CasasBahia|H)', html.split('/')[-1]):
            div = soup.find('div', {'id': 'descricao', 'class': 'descricao'})
            res = div.findAll('p')[1]
            text = res.text
            try:
                price = soup.find('i', {'class': 'sale price'}).text.replace(
                'R$', '').replace(' ', '')
            except AttributeError:
                price = None

        elif re.search('(Multisom|F)', html.split('/')[-1]):
            res = soup.find('div', {'id': 'descricao'})
            text = res.text
            try:
                price = soup.find('p', {'class': 'prices'})
                price = price.find('ins').text.replace(
                    'R$', '').replace(' ', '').replace('Por:', '')
            except AttributeError:
                price = None

        elif re.search('(Made in Brazil|D)', html.split('/')[-1]):
            print('lala')
            res = soup.find('div', {'class': 'infoProd'})
            text = res.text
            try:
                price = soup.find('div', {'class': 'precoPor'}).text.replace(
                    'R$', '').replace(' ', '')
            except AttributeError:
                price = None

        else:
            res = soup.find('div', {'id': 'panCaracteristica',
                                    'class': 'section about especificacion'})
            text = res.text
            try:
                price = soup.find('span', {'id': 'lblPrecoPor', 'class': 'price sale'}).find(
                    'strong').text.replace('R$', '').replace(' ', '').replace('Por:', '')
            except AttributeError:
                price = None

        if type(res) == list():
            text = get_text(text)
        else:
            text = get_text(text)

        characteristcs['Preço'][html] = price

        chars = re.findall(
            '[aA-zZ-áàâãéèêíïóôõöúçñ]*:[(| )][(0-9|)]*[aA-zZ-áàâãéèêíïóôõöúçñ]*', text)

        for char in chars:
            if char.split(': ')[0].title() in characteristcs:
                if char.split(': ')[1] != '':
                    characteristcs[char.split(': ')[0].title()][html] = char.split(': ')[
                        1].title()

    with open('data_specific.json', 'w') as outfile:
        json.dump(characteristcs, outfile)
        outfile.close()


if __name__ == '__main__':
    do_specific([x for x in os.listdir('.') if x.endswith('html')])
