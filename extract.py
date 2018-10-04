import csv
import os
import re
import urllib.request

from bs4 import BeautifulSoup, element
from nltk import word_tokenize

import utils


class Extraction:
    def __init__(self, soups):
        self.soups = soups

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

    def get_tags_for_real(self, coi):
        tgs = coi.find_all(['div', 'p', 'strong', 'span', 'font'])
        tags = list()
        for t in tgs:
            if len(list(t.descendants)) == 1:
                tags.append(t)
        if tgs == []:
            tags.append(coi)
        return tags

    def fix_text(self, text):
        subs = re.sub(r"([A-Z])", r" \1", text).split()
        new_text = ''
        for i, sub in enumerate(subs):
            new_text = new_text + ' ' + sub
        return new_text


if __name__ == '__main__':
    htmls = [x for x in os.listdir('.') if x.endswith('html')]
    texts = []
    for html in htmls:
        extract = Extraction(utils.soups_of_interest(html))
        coi = extract.get_child_of_interest()
        if coi is not None:
            for c in coi(['br']):
                c.extract()
            non_rel = list()
            for child in coi.descendants:
                if not isinstance(child, element.NavigableString):
                    if len(list(child.descendants)) < 1:
                        if not child in non_rel:
                            non_rel.append(child)
            for nr in non_rel:
                for c in coi.find_all(nr.name, nr.attrs):
                    c.extract()
            coi = BeautifulSoup(str(coi).replace('\n<', '<'), 'lxml')
            text = coi.text.replace('; ', '. ').replace(
                ';', '. ').replace('• ', '. ')
            text = extract.fix_text(text)
            texts.append(text)
    texts.sort(key=lambda s: len([i for i in s if i == ':']), reverse=True)
    bigger = texts[0]
    words = word_tokenize(bigger)
    stop_words = list()
    for t in texts[1:]:
        ws = word_tokenize(t)
        for w in ws:
            if w in words:
                stop_words.append(w)
    aux = stop_words
    for i, sw in enumerate(aux):
        stop_words[i] = sw.lower()
    stop_words = list(set(stop_words))
    stop_words = [w for w in stop_words if len(w) >= 5]
    aux = stop_words
    characteristcs = list()
    for t in texts:
        char = list()
        t = t.lower()
        for i, sw in enumerate(stop_words):
            pattern = re.compile(sw+':'+' [aA0-zZ9]*')
            c = re.findall(pattern, t)
            if len(c) > 0:
                char.append(c[-1])
        characteristcs.append(char)
    aux = characteristcs[0]
    cols = list()
    for c in aux:
        cols.append(c.split(':')[0].title())
