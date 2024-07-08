input_size = 2
hidden_size1 = 63
hidden_size2 = 53
output_size = 2

# Parameters between Input and First Hidden Layer
hidden_params1 = (input_size + 1) * hidden_size1
print(f"Parameters between Input and First Hidden Layer: {hidden_params1}")

# Parameters between First and Second Hidden Layer
hidden_params2 = (hidden_size1 + 1) * hidden_size2
print(f"Parameters between First and Second Hidden Layer: {hidden_params2}")

# Parameters between Second Hidden Layer and Output Layer
output_params = (hidden_size2 + 1) * output_size
print(f"Parameters between Second Hidden Layer and Output Layer: {output_params}")

# Total Parameters
total_params = hidden_params1 + hidden_params2 + output_params
print("Total parameters:", total_params)