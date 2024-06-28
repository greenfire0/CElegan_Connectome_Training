# Import the weights dictionary
# Function to count the number of connections
def count_total_entries(weights_dict):
    total_entries = 0
    for inner_dict in weights_dict.values():
        total_entries += len(inner_dict)
    return total_entries

# Main function
if __name__ == "__main__":
    total_entries = count_total_entries(dict)
    print(f"Total number of entries: {total_entries}")