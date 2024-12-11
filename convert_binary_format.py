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

# Example usage:
input_file = 'dirtybookscout.sql'  # Input SQL file containing the binary data
output_file = 'bookscout.sql'  # Output file to save the modified content

process_binary_data(input_file, output_file)

print(f"Processed SQL file saved as {output_file}")