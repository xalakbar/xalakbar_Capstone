import re

def process_binary_data(file_path, output_path):
    with open(file_path, 'r') as file:
        sql_content = file.read()

    processed_content = re.sub(r"X'([0-9A-Fa-f]+)'", r"0x\1", sql_content)

    with open(output_path, 'w') as output_file:
        output_file.write(processed_content)


input_file = 'bookscout.sql'
output_file = 'fixedbookscout.sql'

process_binary_data(input_file, output_file)

print (f"Processed SQL file saved as {output_file}.")