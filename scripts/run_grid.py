"This script runs a grid over different models, datasets, attribution functions, and dimensionality reduction methods. Every combination is tested individually, and every combination result is stored with a Run_ID in outputs. Also, some overall results and tables are stored. This script is the core of the project."
import yaml
with open('configs/grid.yaml', 'r') as file:
    grid_config = yaml.safe_load(file)
with open('configs/base.yaml', 'r') as file:
    base_config = yaml.safe_load(file)

print(base_config)