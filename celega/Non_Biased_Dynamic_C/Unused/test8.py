import pandas as pd

# Load the data from the XLSX file
file_path = 'CElegansNeuronTables.xlsx'  # Replace with your actual file path
df = pd.read_excel(file_path)

# Iterate through the DataFrame to find and swap the necessary entries
for i in range(len(df) - 1):
    # Check if the current row is a GapJunction and the next row is a Send
    if df.at[i, 'Type'] == 'GapJunction' and df.at[i + 1, 'Type'] == 'Send':
        # Check if they have the same Origin and Target
        if df.at[i, 'Origin'] == df.at[i + 1, 'Origin'] and df.at[i, 'Target'] == df.at[i + 1, 'Target']:
            # Swap the rows
            df.iloc[i], df.iloc[i + 1] = df.iloc[i + 1].copy(), df.iloc[i].copy()

# Write the sorted DataFrame back to an XLSX file
output_file_path = 'sorted_' + file_path
df.to_excel(output_file_path, index=False)

# Print the sorted DataFrame to verify the changes
print(df)
