import os

print(os.getcwd())

from cobra.io import read_sbml_model

# Replace 'path_to_your_model.xml' with the actual path to your SBML file
model_path = '../data/e_coli_core.xml'  # Example path
try:
    model = read_sbml_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

