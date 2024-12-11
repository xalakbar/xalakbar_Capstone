import re

def convert_binary_data(match):
    return '0x' + match.group(1).replace("'", "")

with open('bookscout.sql', 'r', encoding='utf-8') as infile:
    content = infile.read()

content = re.sub(r"X'([0-9A-Fa-f]+)'", convert_binary_data, content)

with open('fixedbookscout.sql', 'w', encoding='utf-8') as outfile:
    outfile.write(content)

print("Conversion complete.")