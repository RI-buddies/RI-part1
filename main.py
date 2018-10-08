import json


with open('data_generic.json', 'r') as infile:
    generic = json.load(infile)

with open('data_specific.json', 'r') as infile:
    specific = json.load(infile)
print(specific)

for key in specific:
    char_s = specific[key]
    char_g = generic[key]

