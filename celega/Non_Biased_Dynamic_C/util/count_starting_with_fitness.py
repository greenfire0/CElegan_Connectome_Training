def count_fitness_lines(file_path):
    """Counts the number of lines starting with 'fitness' in a file."""
    try:
        with open(file_path, 'r') as file:
            count = sum(1 for line in file if line.strip().startswith('fitness'))
        print(f"Number of lines starting with 'fitness': {count}")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Replace 'your_file.txt' with the path to your file
file_path = 'f.txt'
count_fitness_lines(file_path)
