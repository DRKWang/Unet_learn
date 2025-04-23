import os


def create_project_structure():
    folders = [
        "./_degenerared",
        "./data/outcome",
        "./data/processed",
        "./data/raw",
        "./docs",
        "./images",
        "./src",
        "./tests/test1",
        "./tests/test2",
        "./tests/test3",
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created: {folder}")

        # Add a placeholder file in each folder
        placeholder_path = os.path.join(folder, ".gitkeep")
        with open(placeholder_path, "w") as f:
            pass  # Empty file
        print(f"  Added placeholder: {placeholder_path}")

    # Create .gitignore
    gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Virtual environments
venv/
env/

# Jupyter Notebook checkpoints
.ipynb_checkpoints/

# IDE settings
.idea/
.vscode/

# Data files
*.csv
*.tsv
*.log
*.h5
*.hdf5

# Output
data/outcome/
data/processed/

# System files
.DS_Store
Thumbs.db
"""
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
        print("Created: .gitignore")

    # Create requirements.txt
    requirements_content = """# compute
numpy<2
pandas

# time
tqdm

# main
opencv-python
protobuf
tokenizers
transformers
pot
torch
torchvision
torchaudio
scikit-learn

# plot
matplotlib
ipywidgets
ipympl

# nb
notebook
jupyter
"""
    with open("requirements.txt", "w") as f:
        f.write(requirements_content)
        print("Created: requirements.txt")


# Run the function
create_project_structure()
