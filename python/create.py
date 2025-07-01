import os
import pydicom
import numpy as np
import re

# Function to check if the dataset exists
def dataset_exists(directory):
    return os.path.exists(directory)

# Valid dataset numbers for each combination of modality and set type
valid_datasets = {
    'CT': {
        'Test_sets': [3, 4, 7, 9, 11, 12, 13, 15, 17, 20, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
        'Train_sets': [1, 2, 5, 6, 8, 10, 14, 16, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    },
    'MR': {
        'Test_sets': [4, 6, 7, 9, 11, 12, 14, 16, 17, 18, 23, 24, 25, 26, 27, 28, 29, 30, 35, 40],
        'Train_sets': [1, 2, 3, 5, 8, 10, 13, 15, 19, 20, 21, 22, 31, 32, 33, 34, 36, 37, 38, 39]
    }
}

# Prompt user for dataset number, modality, and use_set
dataset = input("Choose dataset (CT/MR): ").strip().upper()
while dataset not in ['CT', 'MR']:
    print("\nInvalid entry. Please choose between CT and MR.")
    dataset = input("Choose dataset (CT/MR): ").strip().upper()

# If MR is selected, ask for the specific type (T1DUAL or T2SPIR)
if dataset == 'MR':
    mr_type = input("Choose MR type (T1DUAL/T2SPIR): ").strip().upper()
    while mr_type not in ['T1DUAL', 'T2SPIR']:
        print("\nInvalid entry. Please choose between T1DUAL and T2SPIR.")
        mr_type = input("Choose MR type (T1DUAL/T2SPIR): ").strip().upper()

    # Ask for the specific phase (InPhase or OutPhase) only if T1DUAL is selected
    if mr_type == 'T1DUAL':
        phase_type = input("Choose phase type (InPhase/OutPhase): ").strip()
        while phase_type not in ['InPhase', 'OutPhase']:
            print("\nInvalid entry. Please choose between InPhase and OutPhase.")
            phase_type = input("Choose phase type (InPhase/OutPhase): ").strip()
    else:
        phase_type = ''
else:
    mr_type = ''
    phase_type = ''

# Loop until a valid input for use_set is provided
while True:
    use_set_input = input("\nUse test set (yes) or train set (no)? : ").strip().lower()
    if use_set_input in ['yes', 'no']:
        use_set = use_set_input == 'yes'
        set_type = 'Test_sets' if use_set else 'Train_sets'
        break
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")

# Loop until a valid dataset number is provided
while True:
    try:
        dataset_number = int(input("\nEnter the dataset number: "))
        if dataset_number in valid_datasets[dataset][set_type]:
            break
        else:
            print(f"Invalid dataset number for {dataset} {set_type} set. Please enter a valid number.")
    except ValueError:
        print("Invalid input. Please enter a valid number.")

if dataset == 'CT':
    # Directory containing the DICOM files
    dicom_directory = f'python/{set_type}/{dataset}/{dataset_number}/DICOM_anon/'
else:
    if mr_type == 'T1DUAL':
        dicom_directory = f'python/{set_type}/{dataset}/{dataset_number}/{mr_type}/DICOM_anon/{phase_type}'
    else:
        dicom_directory = f'python/{set_type}/{dataset}/{dataset_number}/{mr_type}/DICOM_anon/'

# Check if the dataset exists, if not prompt the user to re-enter the dataset number
while not dataset_exists(dicom_directory):
    print(dicom_directory)

    print(f"Dataset number {dataset_number} not found in {dataset} {set_type} set.")
    while True:
        try:
            dataset_number = int(input("Enter the dataset number: "))
            if dataset_number in valid_datasets[dataset][set_type]:
                break
            else:
                print(f"Invalid dataset number for {dataset} {set_type} set. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    dicom_directory = f'{dataset}/{dataset_number}/DICOM_anon/'

print(dicom_directory)

# Function to read DICOM files into a single 3D numpy array
def read_dicom_files(directory):
    dicom_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".dcm"):
            filepath = os.path.join(directory, filename)
            dicom_data = pydicom.dcmread(filepath)
            pixel_array = dicom_data.pixel_array.astype(np.float32) 
            dicom_files.append(pixel_array)
    dicom_data = np.array(dicom_files)
    print(f"Shape of the 3D DICOM data: {dicom_data.shape}")
    return dicom_data

# Function to convert DICOM files to binary and save the shape
def dicom_to_binary(dicom_directory, binary_file_path, shape_file_path):
    dicom_data = read_dicom_files(dicom_directory)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(binary_file_path), exist_ok=True)
    
    dicom_data.tofile(binary_file_path)
    with open(shape_file_path, 'w') as f:
        f.write(','.join(map(str, dicom_data.shape)))
    print(f"Data saved to {binary_file_path} and shape saved to {shape_file_path}")

# Extract the number from the directory path
number = re.search(r'/(\d+)/', dicom_directory).group(1)

# Paths to the output binary and shape files
if dataset == 'MR':
    if mr_type == 'T1DUAL':
        binary_file_path = f'data/inputs/{number}_{dataset}_{mr_type}_{phase_type}.bin'
        shape_file_path = f'data/inputs/{number}_{dataset}_{mr_type}_{phase_type}_shape.txt'
    else:
        binary_file_path = f'data/inputs/{number}_{dataset}_{mr_type}.bin'
        shape_file_path = f'data/inputs/{number}_{dataset}_{mr_type}_shape.txt'
else:
    binary_file_path = f'data/inputs/{number}_{dataset}.bin'
    shape_file_path = f'data/inputs/{number}_{dataset}_shape.txt'

# Convert DICOM files to binary and save shape
dicom_to_binary(dicom_directory, binary_file_path, shape_file_path)