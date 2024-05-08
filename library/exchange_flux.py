import cobra
from cobra.io import read_sbml_model  # Import to load SBML models

def write_model_details_to_file(sbml_path, output_dir):
    # Extract the model name from the SBML file path
    model_name = sbml_path.split('/')[-1].replace('.xml', '')

    # Load the model from the SBML file
    model = read_sbml_model(sbml_path)

    # Prepare the output file path
    output_file_path = f'{output_dir}/{model_name}_model_details.txt'

    # Open a file to write
    with open(output_file_path, 'w') as file:
        # Write the number of exchange fluxes
        exchange_reactions = model.exchanges
        file.write(f"Number of Exchange Fluxes: {len(exchange_reactions)}\n")

        # Write exchange reactions details
        file.write("\nExchange Fluxes in the Model:\n")
        for reaction in exchange_reactions:
            file.write(f"ID: {reaction.id}, Name: {reaction.name}, Reaction: {reaction.reaction}, "
                       f"Lower bound: {reaction.lower_bound}, Upper bound: {reaction.upper_bound}\n")
        
        # Write the number of total reactions
        total_reactions = model.reactions
        file.write(f"\nNumber of Total Reactions: {len(total_reactions)}\n")

        # Write all reactions details
        file.write("\nAll Reactions in the Model:\n")
        for reaction in total_reactions:
            file.write(f"{reaction.id}: {reaction.name}\n")

# Specify the path to your SBML file and the output directory
sbml_file_path = 'data/Alteromonas_Model.xml'  # Ensure this is the correct local path
output_directory = 'data/model_details'

# Call the function with the path to your SBML file and output directory
write_model_details_to_file(sbml_file_path, output_directory)
