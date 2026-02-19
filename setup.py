from cx_Freeze import setup, Executable
import os

# Base setting (None for console app)
base = None

# Define the script to convert to .exe (rename as needed)
executables = [
    Executable("mainV2.py", base=base)  # Replace with your main script name
]

# Required packages used in your code
packages = [
    "os", "sys", "json", "random", "datetime", "csv",
    "torch", "torchvision", "numpy", "PIL", "matplotlib",
    "pandas", "tqdm"
]

# Include external files and folders
include_files = [
    "config.json",  # Configuration file
    "model",        # Folder containing ObjModel
    "utils"         # Folder containing util.py, ObjDataset.py
]

setup(
    name="DotMatrixTrainer",
    version="0.1",
    description="Trainer for DotMatrix object detection",
    executables=executables,
    options={
        "build_exe": {
            "packages": packages,
            "include_files": include_files,
            "excludes": ["tkinter"],  # Optional: exclude unused modules
        }
    }
)
