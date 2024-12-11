import re

# Function to prepend 0x and remove single quotes and X prefix around binary data
def process_binary_data(file_path, output_path):
    # Read the content of the SQL file
    with open(file_path, 'r', encoding='utf-8') as file:
        sql_content = file.read()

    # Use regular expression to find binary data patterns like X'...' and replace with 0x...
    processed_content = re.sub(r"X'([0-9A-Fa-f]+)'", r"0x\1", sql_content)

    # Write the modified content to a new SQL file
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write(processed_content)

input_file = 'bookscout.sql'  # Input SQL file containing the binary data
output_file = 'fixedbookscout.sql'  # Output file to save the modified content

process_binary_data(input_file, output_file)

print(f"Processed SQL file saved as {output_file}")