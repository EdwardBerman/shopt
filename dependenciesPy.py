import subprocess

def install_packages(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            package = line.strip()
            if package:
                try:
                    subprocess.check_call(['pip', 'install', package])
                    print(f"Successfully installed {package}")
                except subprocess.CalledProcessError as e:
                    print(f"Error installing {package}: {e}")

# Provide the path to your text file containing package names
file_path = 'packages.txt'
install_packages(file_path)

