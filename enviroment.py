import datetime
from pathlib import Path
import shutil
import os
import json
import pickle
class Environment:

    @classmethod
    def load(cls, filepath):
        # Ensure the filepath is a Path object
        filepath = Path(filepath)

        # Check if the file exists before attempting to open
        if not filepath.exists():
            raise FileNotFoundError(f"The file {filepath} does not exist.")

        # Deserialize the Environment object from the file
        with open(filepath, "rb") as file:
            environment = pickle.load(file)

        # Ensure the deserialized object is an instance of the class
        if not isinstance(environment, cls):
            raise TypeError(f"The file {filepath} does not contain a valid {cls.__name__} object.")

        return environment
    def __init__(self, environment_name="env"):
        prefix = datetime.datetime.now().strftime("%Y%m%d-%H")
        self.base_path = Path("ExtractionEnvironments") / f"{prefix}-{environment_name}"
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Create directories under the base path and store their paths
        self.dirs = {}  # Dictionary to store directory names and their paths
        self.dirs["00-VanillaComponents"] = self.base_path / "00-VanillaComponents"
        self.dirs["01-AllComponents"] = self.base_path / "01-AllComponents"
        self.dirs["02-Logs"] = self.base_path / "02-Logs"
        self.dirs["03-GH_Files"] = self.base_path / "03-GH_Files"
        self.dirs["04-CSV"] = self.base_path / "04-CSV"
        self.dirs["05-GraphML"] = self.base_path / "05-GraphML"
        self.dirs["99-GH-Lib"] = self.base_path / "99-GH-Lib"
        self.gh_path =  r"C:\Users\jossi\AppData\Roaming\Grasshopper\Libraries"


        # Create the directories
        for path in self.dirs.values():
            path.mkdir(parents=True, exist_ok=True)

        config_dir = Path("ExtractionEnvironments")
        config_dir.mkdir(parents=True, exist_ok=True)

        config = {'dirs': {dir_name: str(dir_path) for dir_name, dir_path in self.dirs.items()}}

        with open(config_dir / 'config.json', 'w') as f:
            json.dump(config, f)

        # Assuming self.base_path is defined and is a Path object
        file_path = self.base_path / "environment.pkl"

        # Serializing the object
        if not file_path.exists():
            with open(file_path, "wb") as file:
                pickle.dump(self, file)


    def clone_env(self, clone=False):
        GH_path =self.gh_path
        if clone:
            for root, dirs, files in os.walk(GH_path):
                for file in files:
                    source_path = Path(root) / file
                    destination_path = self.dirs["99-GH-Lib"] / file
                    shutil.copy(source_path, destination_path)

    def reinstate_env(self, override = False):
        if input("Are you sure you want to reinstate the environment? \n"
                 "This will replace existing files in GH Libraries Folder"
                 "with those belonging to this environment? [y/n]").lower() == "y"\
                or override:
            print("Reinstating environment...")
            GH_path = self.gh_path
            lib_path = self.dirs["99-GH-Lib"]

            # Delete all files in GH_path
            for root, dirs, files in os.walk(GH_path):
                for file in files:
                    file_path = Path(root) / file
                    if file_path.is_file():
                        file_path.unlink()

            # Copy all files from 99-GH-Lib back to GH_path
            for file in lib_path.iterdir():
                if file.is_file():
                    shutil.copy(file, GH_path)
        else:
            print("Reinstate cancelled")

    def view_gh_environment(self):
        for root, dirs, files in os.walk(self.dirs["99-GH-Lib"]):
            for file in files:
                print(file)

    def copy_file(self, src, dst):
        """
        Copies a file from the source path to the destination path.

        :param src: The source file path as a string or Path object.
        :param dst: The destination file path as a string or Path object.
        """
        # Ensure that the source exists and is a file
        if not src.exists() or not src.is_file():
            print(f"Source file does not exist or is not a file: {src}")
            return

        # Ensure the destination directory exists, if not, create it
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Copy the file
        try:
            shutil.copyfile(src, dst)
            print(f"File copied successfully from {src} to {dst}.")
        except Exception as e:
            print(f"Error copying file from {src} to {dst}: {e}")

    def get_environment(self, vanilla_components:Path =None,  clone_env = True):
        print("Setting environment variables")
        print("Copying vanilla components")
        if not vanilla_components:
            vanilla_components = Path(r"C:\Users\jossi\Dropbox\Office_Work\Jos\GH_Graph_Learning\Grasshopper Components\240307-CoreComponents\grasshopper_components.csv")
        if vanilla_components: self.copy_file(vanilla_components, self.dirs["00-VanillaComponents"])
        print("Copying components")
        self.clone_env(clone_env)




