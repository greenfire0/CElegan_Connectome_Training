import numpy as np
import pandas as pd

def write_array_to_file(array, filename):
    try:
        with open(filename, 'w') as file:
            for item in array:
                file.write(f"{item}\n")
        print(f"Array successfully written to {filename}")
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")


def read_array_from_file(filename):
    try:
        with open(filename, 'r') as file:
            array = [float(line.strip()) for line in file]
        print(f"Array successfully read from {filename}")
        return array
    except Exception as e:
        print(f"An error occurred while reading from the file: {e}")
        return []
    

def read_arrays_from_csv_pandas(filename: str): 
    df = (pd.read_csv(filename, header=None))
    print(f"{(df.shape[0])} Worms Loaded")
    arrays = df.values.tolist()  
    assert len(df) == len(arrays)
    return arrays

def delete_arrays_csv_if_exists():
    import os
    filename = 'arrays.csv'
    if os.path.exists(filename):
        os.remove(filename)
        print(f"{filename} has been deleted.")
    else:
        print(f"{filename} does not exist.")
    image_folder = '/home/miles2/Escritorio/C.-Elegan-bias-Exploration/celega/Non_Biased_Dynamic_C/tmp_img'
    for img in sorted(os.listdir(image_folder)):
        if img.endswith('.png'):
            img_path = os.path.join(image_folder, img)
            os.remove(img_path)
            print(f"Deleted image: {img_path}")


def save_last_100_rows(input_file: str, output_file: str):
    # Read the CSV file
    df = read_arrays_from_csv_pandas(input_file)
    
    # Get the last 100 rows
    last_100_rows = df[len(df)-100:len(df)]
    last_100_rows=(pd.DataFrame(last_100_rows))
    # Save the last 100 rows to a new CSV file
    last_100_rows.to_csv(output_file, index=False,header=False)
    print(last_100_rows.head(5))
    print(f"Saved the last 100 rows to {output_file}")


if 0:
    input_file = '/home/miles2/Escritorio/C.-Elegan-bias-Exploration/14:53_genetic_tri_550.csv'    # Replace with your input file name
    output_file = 'output5_genetic_tri.csv'  # Replace with your desired output file name
    save_last_100_rows(input_file, output_file)

def read_excel(file_path):
    df = pd.read_excel(file_path, sheet_name='Connectome')
    return df.values.tolist()

# Flatten the dictionary values into a list
def flatten_dict_values(d):
    flattened = []
    for key, subdict in d.items():
        for subkey, value in subdict.items():
            flattened.append((subkey, value,key))
    return flattened

def retreive_nums(d):
    values_list=[]
    #print(d)
    for sub_dict in d:
        values_list.append(sub_dict[1])
    values_list=np.array(values_list,dtype=object)
    return values_list

def populate_dict_values(d, values_list):
    index = 0
    for key, subdict in d.items():
        for subkey in subdict.keys():
            subdict[subkey] = values_list[index]
            index += 1
    return d

def convert_weight_matrix(weight_matrix):
    """
    Converts a list of lists where each sublist may have 1 or 2 elements into a NumPy array of float64.
    
    Parameters:
        weight_matrix (list of lists): The input matrix with sequences of 1 or 2 elements.
    
    Returns:
        np.ndarray: A NumPy array of float64 with all elements properly converted.
    
    Raises:
        ValueError: If any sublist contains more than 1 element or if conversion to float fails.
    """
    # Initialize a list to hold the converted values
    converted_values = []

    # Iterate over each sublist in the weight_matrix
    for sublist in weight_matrix:
        if len(sublist) > 2:
            raise ValueError(f"Each sublist should have at most 2 elements, but got: {sublist}")
        
        # Convert each element in the sublist to float
        for item in sublist:
            try:
                converted_values.append(float(item))
            except ValueError:
                raise ValueError(f"Cannot convert item to float: {item}")

    # Convert the list of values to a NumPy array of float64
    return np.array(converted_values, dtype=np.float64)