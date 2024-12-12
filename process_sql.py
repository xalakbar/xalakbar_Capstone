import re

def process_binary_data(file_path, output_path):
    # Open the file with UTF-16 LE encoding to read it properly
    with open(file_path, 'r', encoding='utf-16') as file:
        file_content = file.read()

    # Use regex to match binary patterns (X'...' -> 0x...)
    processed_content = re.sub(r"X'([0-9A-Fa-f]+)'", r"0x\1", file_content)

    # Write the processed content back to a new file, using UTF-16 LE encoding
    with open(output_path, 'w', encoding='utf-16') as output_file:
        output_file.write(processed_content)

    print(f"Processed SQL file saved as {output_path}")


def process_user_data(content):
    # Use regex to match and modify the INSERT INTO users statement
    content = re.sub(r"INSERT INTO users VALUES\((\d+),\s*'([^']+)',\s*'([^']+)'\);", 
                     r"INSERT INTO users VALUES('\2', '\3');", content)
    return content

input_file = 'dirtybookscout.sql' 
output_file = 'bookscout.sql' 

process_binary_data(input_file, output_file)