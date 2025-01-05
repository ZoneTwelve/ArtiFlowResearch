import json

def process_dataset(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line.strip())  # Assuming one JSON object per line
            title = data.get('title', '')
            
            # Filter titles containing '$title' or empty strings
            if '$title' not in title and title != '':
                # Wrap the title with <title> and </title>
                data['title'] = f"<title>{title}</title>"
                json.dump(data, outfile)
                outfile.write('\n')

# Example usage
input_file = 'example_gen.output.jsonl.b'
output_file = 'processed_dataset.json'
process_dataset(input_file, output_file)

