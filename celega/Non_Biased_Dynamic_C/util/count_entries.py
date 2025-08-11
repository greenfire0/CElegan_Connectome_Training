# Import the weights dictionary
from Worm_Env.weight_dict import dict as weights_dict, all_neuron_names,muscleList
import pandas as pd


EXCLUDE_PREFIXES = ('MVU', 'MVL', 'MDL', 'MVR', 'MDR')


connections = 0
totalcon = 0
filtered_neurons = [n for n in all_neuron_names if not n.startswith(EXCLUDE_PREFIXES)]
filtered_muscles = [n for n in all_neuron_names if n.startswith(EXCLUDE_PREFIXES)]

for neuron in filtered_neurons:
    if neuron in weights_dict:
        connections += len(weights_dict[neuron])

for neuron in all_neuron_names:
    if neuron in weights_dict:
        totalcon += len(weights_dict[neuron])


print(len(filtered_neurons), connections,totalcon)


# Main function

# --- CONFIG -----------------------------------------------------------
XLS_FILE   = "CElegansNeuronTables.xlsx"   # adjust the path if needed
ORIGIN_COL = "Origin"
TARGET_COL = "Target"
# ----------------------------------------------------------------------

df = pd.read_excel(XLS_FILE)

# Clean up whitespace / NaNs and convert to Python sets
origins  = set(df[ORIGIN_COL].dropna().str.strip())
targets  = set(df[TARGET_COL].dropna().str.strip())

# Union gives the full neuron list
all_neurons = origins | targets

print(f"Unique origin neurons : {len(origins)}")
print(f"Unique target neurons : {len(targets)}")
print(f"Total unique neurons  : {len(all_neurons)}")

# Uncomment below if you want to see the actual names
# print(sorted(all_neurons))

print((set(filtered_neurons)-all_neurons))

print(len(muscleList))