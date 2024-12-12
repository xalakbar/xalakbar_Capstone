import re

def process_data(file_path, output_path):
    # Open the file with UTF-16 LE encoding to read it properly
    with open(file_path, 'r', encoding='utf-16') as file:
        file_content = file.read()

    # Use regex to match binary patterns (X'...' -> 0x...)
    file_content = re.sub(r"X'([0-9A-Fa-f]+)'", r"0x\1", file_content)

    file_content = re.sub(r"INSERT INTO users VALUES\((\d+),\s*'([^']+)',\s*'([^']+)'\);", 
                     r"INSERT INTO users VALUES('\2', '\3');", file_content)

    # Write the processed content back to a new file, using UTF-16 LE encoding
    with open(output_path, 'w', encoding='utf-16') as output_file:
        output_file.write(file_content)

    print(f"Processed SQL file saved as {output_path}")

input_file = 'dirtybookscout.sql' 
output_file = 'bookscout.sql' 

process_data(input_file, output_file)