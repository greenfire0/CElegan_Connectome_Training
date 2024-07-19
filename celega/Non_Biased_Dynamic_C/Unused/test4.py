from PyNomad import *

# Define the objective function for NOMAD
def objective_function(x):
    return x[0]**2 + x[1]**2

# Initialize NOMAD with the objective function and bounds
nomad = Nomad(obj_func=objective_function, lb=[-5.0, -5.0], ub=[5.0, 5.0])

# Set NOMAD parameters
nomad.set_parameter(name='MAX_BB_EVAL', value=1000)  # Maximum number of function evaluations

# Run NOMAD optimization
solution = nomad.optimize()

# Retrieve and print results
best_solution = solution.get_final_x()
best_objective_value = solution.get_final_f()

print("Best solution found:", best_solution)
print("Objective value at best solution:", best_objective_value)
