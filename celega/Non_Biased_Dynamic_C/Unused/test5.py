import PyNomad 
import inspect

# Get the file location of the numpy module
numpy_location = inspect.getfile(PyNomad)
print(numpy_location) 