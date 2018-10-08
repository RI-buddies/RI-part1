import csv
import os
import re
import json
import urllib.request

from bs4 import BeautifulSoup, element
from nltk import word_tokenize

import utils


class Extraction:

    soups = list()
    texts = dict()
    coi = str()
    stop_words = list()
    characteristcs = list()
    cols = dict()
    chars = {'Acabamento': list(), 'Fundo': list(),
             'Cordas': list(), 'Braço': list(), 'Preço': list()}

    def __init__(self, htmls):
        self.htmls = htmls

    def get_soups(self):
        return self.soups

    def get_soups_texts(self):
        texts = list()
        for soup in self.soups:
            texts.append(soup.text)
        return texts

    def get_child_of_interest(self):
        coi = list()
        for soup in self.soups:
            interest = soup.find_all(['div', 'p'])
            awesome = list()
            for i in interest:
                try:
                    if re.search(re.compile('(Braço|braço|Corpo|corpo)'), i.text):
                        awesome.append(i)
                except TypeError:
                    pass

            if len(awesome) > 0:
                s = awesome[-1]
                if not s in coi:
                    if len(s.text) < 30:
                        s = s.parent
                    coi.append(s)
            else:
                if re.search(re.compile('(Braço|braço|Cordas|cordas)'), soup.text):
                    coi.append(soup)
        if len(coi) > 0:
            return coi[-1]
        else:
            return None

    def get_text(self, child=None):
        if child is None:
            child = self.coi
        text = child.text.replace('; ', '. ').replace(
            ';', '. ').replace('• ', '. ')
        subs = re.sub(r"([A-Z])", r" \1", text).split()
        new_text = ''
        for i, sub in enumerate(subs):
            new_text = new_text + ' ' + sub
        return new_text

    def treat_coi(self):
        if self.coi is not None:
            for c in self.coi(['br']):
                c.extract()
            non_rel = list()
            for child in self.coi.descendants:
                if not isinstance(child, element.NavigableString):
                    if len(list(child.descendants)) < 1:
                        if not child in non_rel:
                            non_rel.append(child)
            for nr in non_rel:
                for c in self.coi.find_all(nr.name, nr.attrs):
                    c.extract()
            self.coi = BeautifulSoup(str(self.coi).replace('\n<', '<'), 'lxml')

    def through_htmls(self):
        for html in self.htmls:
            self.get_price(html)
            self.soups = utils.soups_of_interest(html)
            self.coi = extract.get_child_of_interest()
            self.treat_coi()
            text = self.get_text()
            self.texts[html] = text

    def get_stop_words(self):
        self.texts = sorted(self.texts.items(), key=lambda h: len(
            re.findall(':', h[1])), reverse=True)
        bigger = self.texts[0]
        words = word_tokenize(bigger[1])
        for _, t in self.texts[1:]:
            ws = word_tokenize(t)
            for w in ws:
                if w in words:
                    self.stop_words.append(w)
        aux = self.stop_words
        for i, sw in enumerate(aux):
            self.stop_words[i] = sw.lower()
        self.stop_words = list(set(self.stop_words))
        self.stop_words = [w for w in self.stop_words if len(w) >= 5]
        return self.stop_words

    def get_characteristcs(self):
        for html, t in self.texts:
            char = list()
            t = t.lower()
            for _, sw in enumerate(self.stop_words):
                pattern = re.compile(sw+':'+' [aA0-zZ9]*')
                c = re.findall(pattern, t)
                if len(c) > 0:
                    char.append(c[-1])
            self.characteristcs.append((html, char))
        return self.characteristcs

    def get_cols(self):
        aux = self.characteristcs[0][1]
        for c in aux:
            self.cols[c.split(':')[0].title()] = list()
        for x in self.characteristcs:
            for c in x[1]:
                tipo = c.split(': ')[0].title()
                char = c.split(': ')[1].title()
                if tipo in self.cols:
                    self.cols[tipo].append((char, x[0]))
        return self.cols

    def get_real_chars(self, cols):
        for col in cols:
            if col in self.chars:
                self.chars[col] = cols[col]
        return self.chars

    def get_price(self, html):
        sauce = open(html, 'r')
        s = BeautifulSoup(sauce, 'lxml')
        price = re.findall(
            '([0-9][0-9][0-9],[0-9][0-9]|[0-9][(.|)][0-9][0-9][0-9],[0-9][0-9])', str(s.text))[0]
        self.chars['Preço'].append((price, html))

    def execute(self):
        self.through_htmls()
        self.get_stop_words()
        self.get_characteristcs()
        return self.get_real_chars(self.get_cols())


if __name__ == '__main__':
    htmls = [x for x in os.listdir('.') if x.endswith('html')]
    extract = Extraction(htmls)
    char = extract.execute()
    with open('data_generic.json', 'w') as outfile:
        json.dump(char, outfile)
#Acabamento, Fundo, Cordas, Braço, Preço
