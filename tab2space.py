import os
import json


def replace_tabs_in_json(folder_path):
    # List all files in the given folder
    files = os.listdir(folder_path)

    # Filter out only JSON files
    json_files = [file for file in files if file.endswith('.json')]

    for json_file in json_files:
        # Construct full file path
        file_path = os.path.join(folder_path, json_file)

        # Read the JSON data
        with open(file_path, 'r', encoding='utf-8') as file:
            # Load JSON data, replacing tabs with spaces
            content = json.load(file)

        # Convert JSON object back to string, replacing tabs with spaces
        # in the process of serialization
        modified_json_string = json.dumps(content, indent=4, ensure_ascii=False).replace('\t', '    ')

        # Write the modified content back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(modified_json_string)


# Replace 'path_to_folder' with the path to your folder containing JSON files
replace_tabs_in_json("data/synthetic/panda_synth_train_dr")
