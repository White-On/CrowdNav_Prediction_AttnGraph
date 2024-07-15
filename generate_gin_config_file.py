import os
import itertools

def clear_old_configs(output_dir):
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            os.remove(file_path)
        print(f"Removed old config files in {output_dir}")

def read_config(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def parse_values(line):
    if '[' in line and ']' in line:
        key, values = line.split('=')
        values = values.strip().strip('[]').split(',')
        values = [v.strip() for v in values]
        return key.strip(), values
    return None, None

def generate_configs(base_config, output_dir):
    variable_lines = []
    fixed_lines = []
    
    for line in base_config:
        key, values = parse_values(line)
        if key:
            variable_lines.append((key, values))
        else:
            fixed_lines.append(line)
    
    combinations = list(itertools.product(*[values for _, values in variable_lines]))
    
    for i, combo in enumerate(combinations):
        config_lines = fixed_lines[:]
        for (key, _), value in zip(variable_lines, combo):
            config_lines.append(f"{key} = {value}\n")
        
        output_path = os.path.join(output_dir, f"config_{i+1}.gin")
        with open(output_path, 'w') as file:
            file.writelines(config_lines)
        print(f"Generated {output_path}")

def main():
    base_config_path = 'config.gin'  # Chemin vers le fichier de configuration de base
    output_dir = 'gin_config_files'  # Répertoire de sortie pour les fichiers générés
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    clear_old_configs(output_dir)
    
    base_config = read_config(base_config_path)
    generate_configs(base_config, output_dir)

if __name__ == "__main__":
    main()