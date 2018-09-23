import os
import re
import urllib.request

from bs4 import BeautifulSoup

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

    def get_children_of_interest(self):
        coi = list()
        for soup in self.soups:
            interest = soup.find_all(['div','p'])
            awesome = list()
            for i in interest:
                try:
                    if re.search(re.compile('(Corda|corda)'), i.text):
                        awesome.append(i)
                except TypeError:
                    pass
            if len(awesome)>0:      
                if not awesome[-1] in coi:
                    coi.append(awesome[-1])
        return coi
                
if __name__ == '__main__':
    htmls = [x for x in os.listdir('.') if x.endswith('html')]
    for html in htmls:
        print('\n----------------------------------------------------------------------------------------------\n', html, ':')
        extract = Extraction(utils.soups_of_interest(html))
        print(extract.get_children_of_interest())
