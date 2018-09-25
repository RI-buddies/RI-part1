import os
import re
import urllib.request

from bs4 import BeautifulSoup, element

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
                    if re.search(re.compile('(Braço|braço)'), i.text):
                        awesome.append(i)
                except TypeError:
                    pass
            if len(awesome) > 0:
                s = awesome[-1]
                if not s in coi:
                    if len(s.text) < 30:
                        s = s.parent
                    coi.append(s)
        return coi[-1]

    def get_tags_for_real(self, coi):
        tgs = coi.find_all(['div', 'p', 'strong', 'span', 'font'])
        tags = list()
        for t in tgs:
            if len(list(t.descendants)) == 1:
                tags.append(t)
        if tgs == []:
            tags.append(coi)
        print(tags)
        return tags

if __name__ == '__main__':
    htmls = [x for x in os.listdir('.') if x.endswith('html')]
    for html in htmls:
        print('\n----------------------------------------------------------------------------------------------\n', html, ':')
        extract = Extraction(utils.soups_of_interest(html))
        coi = extract.get_child_of_interest()
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
        coi = BeautifulSoup(str(coi).replace('\n<','<'),'lxml')
        print (coi)
        extract.get_tags_for_real(coi)
