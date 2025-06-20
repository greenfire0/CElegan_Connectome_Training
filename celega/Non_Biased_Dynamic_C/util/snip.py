import pandas as pd
import os 
from Worm_Env.connectome import WormConnectome
import csv
from pathlib import Path

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
    
def write_worm_to_csv(base_name: str, worm: "WormConnectome", max_rows: int = 100) -> None:
    """
    Appends the worm’s weight matrix to a CSV file.
    If the target file already has `max_rows` rows, it rolls over to a new
    file by appending “+1”, “+2”, … to the base name.

    Parameters
    ----------
    base_name : str
        The filename **without** extension (e.g. "worms").
    worm : WormConnectome
        Object holding .weight_matrix (NumPy array-like).
    max_rows : int, optional
        Maximum rows allowed per file before rollover, default = 100.
    """
    # Find the first file with < max_rows rows (or an empty new one).
    idx = 0
    while True:
        fname = Path(f"{base_name}{f'{idx}' if idx else ''}.csv")
        if not fname.exists():
            break                       # fresh file – safe to use
        with fname.open("r", newline="") as f:
            rows = sum(1 for _ in f)
        if rows < max_rows:
            break                       # current file has space
        idx += 1                        # otherwise try next suffix

    # Append the worm matrix to the selected file.
    with fname.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(worm.weight_matrix.tolist())

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



def save_last_100_rows(input_file: str, output_file: str):
    # Read the CSV file
    interval = 10
    df = read_arrays_from_csv_pandas(input_file)
    start = (len(df)-len(df)%interval)
    print(start)
    while start>=interval:
        last_100_rows = df[start-interval:start]
        start-=interval
        last_100_rows=(pd.DataFrame(last_100_rows))
        last_100_rows.to_csv(output_file+str(start)+".csv", index=False,header=False)
        print(f"Saved the last 100 rows to {output_file}")

def read_excel(file_path):
    df = pd.read_excel(file_path, sheet_name='Connectome')
    return df.values.tolist()

def flatten_dict_values(d):
    flattened = []
    for key, subdict in d.items():
        for subkey, value in subdict.items():
            flattened.append((subkey, value,key))
    return flattened

def read_last_array_from_csv(csv_file):
    df = pd.read_csv(csv_file, header=None)
    last_array = df.iloc[-1].to_numpy()
    return last_array

if 0: ## not sure what this garabge is but it sux

    ##this is used for breakiung appart csv files
    base_dir = os.path.dirname(__file__)  # Get the directory of the current script
    full_folder_path = os.path.join(base_dir)
    input_file = os.path.join(full_folder_path, "arrays.csv")
    output_file = '250-sq-NO'  # Replace with your desired output file name
    save_last_100_rows(input_file, output_file)