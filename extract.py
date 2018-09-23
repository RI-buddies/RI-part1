import os
import re
import urllib.request

from bs4 import BeautifulSoup

import utils


htmls = [x for x in os.listdir('.') if x.endswith('html')]
paragraphs = []
for html in htmls:
    for ip in utils.soups_of_interest(html):
        print(ip, '\n---------------------\n')
