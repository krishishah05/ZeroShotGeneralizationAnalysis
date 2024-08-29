import os

# Specify the directory where the files are located
directory = 'Pretraining&TrainingDatasets'  # Change this to your directory path

# Loop through all files in the directory
for filename in os.listdir(directory):
    # Construct the full file path
    file_path = os.path.join(directory, filename)
    
    # Skip if it's not a file
    if not os.path.isfile(file_path):
        continue
    
    # Get the file name without the extension
    base_name, _ = os.path.splitext(filename)
    
    # Create the new file name with a .txt extension
    new_file_name = base_name + '.txt'
    
    # Construct the full new file path
    new_file_path = os.path.join(directory, new_file_name)
    
    # Rename the file
    os.rename(file_path, new_file_path)

    print(f'Renamed: {filename} -> {new_file_name}')
