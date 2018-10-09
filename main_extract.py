import json
import os

from extract import do_generic
from extract_specific import do_specific

def get_metrics(letter):
    with open('data_generic_'+letter+'.json', 'r') as infile:
        generic = json.load(infile)

    with open('data_specific_'+letter+'.json', 'r') as infile:
        specific = json.load(infile)

    n = 0
    for key in specific:
        for html in specific[key]:
            if(html.split('/')[-1][0] == letter):   
                n += 1

    e = 0
    for key in generic:
        for html in generic[key]:
            if(html.split('/')[-1][0] == letter):
                e += 1

    c = 0
    for key in specific:
        for html in specific[key]:
            if html in generic[key]:
                if generic[key][html] == specific[key][html]:
                    c += 1

    recall = c/n
    precision = c/e
    fmeasure = 2 * recall * precision / (recall + precision)
    return recall, precision, fmeasure


def main():
    path = os.path.join(os.path.abspath('.'),'Classifier/data/html')
    for letter in ['A', 'B', 'C', 'D', 'F', 'G', 'H']:
        do_specific([os.path.join(path, x) for x in os.listdir('./Classifier/data/html') if x.endswith('p.html') and x.startswith(letter)])
        do_generic([os.path.join(path, x) for x in os.listdir('./Classifier/data/html') if x.endswith('p.html') and x.startswith(letter)])
        metrics = get_metrics(letter)
        print(letter,':\n',metrics)

if __name__ == '__main__':
    main()
