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
    import pandas as pd
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
    import pandas as pd
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
    import pandas as pd
    df = pd.read_excel(file_path, sheet_name='Connectome')
    return df.values.tolist()

# Flatten the dictionary values into a list
def flatten_dict_values(d):
    flattened = []
    for key, subdict in d.items():
        for subkey, value in subdict.items():
            flattened.append((subkey, value))
    return flattened
