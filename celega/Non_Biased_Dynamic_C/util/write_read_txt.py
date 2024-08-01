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
    df.to_csv(filename, index=False, header=False)
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