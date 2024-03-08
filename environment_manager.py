import datetime
import json
import os
import pickle
import shutil
from pathlib import Path
from gh_logging import LogManager


class EnvironmentManager:
    _environments = {}  # Stores Environment instances keyed by environment_name

    # Define directory structure as a class-level attribute.
    DIR_STRUCTURE = {
        "vanilla": "00-VanillaComponents",
        "all": "01-AllComponents",
        "logs": "02-Logs",
        "files": "03-GH_Files",
        "csv": "04-CSV",
        "graphml": "05-GraphML",
        "ghlib": "99-GH-Lib"
    }

    GHPATH = r"C:\Users\jossi\AppData\Roaming\Grasshopper\Libraries"
    VANILLA_PATH = r"C:\Users\jossi\Dropbox\Office_Work\Jos\GH_Graph_Learning\Grasshopper Components\240307-CoreComponents\vanilla_components.csv"
    GH_FILE_PATH = r"C:\Users\jossi\Dropbox\Office_Work\Jos\GH_Graph_Learning\GHData"
    class Environment:
        def __init__(self, environment_name):
            prefix = datetime.datetime.now().strftime("%y%m%d")
            self.environment_name = environment_name
            self.base_path = Path("ExtractionEnvironments") / f"{prefix}-{environment_name}"
            self.base_path.mkdir(parents=True, exist_ok=True)
            self.logmanager = None
            # Initialize directories based on the class-level DIR_STRUCTURE
            self.dirs = {name: self.base_path / dirname for name, dirname in EnvironmentManager.DIR_STRUCTURE.items()}
            for path in self.dirs.values():
                path.mkdir(parents=True, exist_ok=True)
            #  Create loggers.
            LogManager.new_logger("clog", self.dirs['logs'], "clog.log")
            LogManager.new_logger("flog", self.dirs['logs'], 'flog.log')
            self.gh_path = Path(EnvironmentManager.GHPATH)
            self.initialise()
            self.serialize()

        def serialize(self):
            file_path = self.base_path / "environment.pkl"
            with open(file_path, "wb") as file:
                pickle.dump(self, file)

        def clone_env(self, clone=False):
            GH_path = Path(self.gh_path)  # Ensure GH_path is a Path object
            lib_path = self.dirs.get('ghlib')
            r, d, f = os.walk(lib_path)
            # only clone if there isnt already a cloned environment
            if clone and len(f) == 0:
                # os.walk() iterates over the directories and files in GH_path
                for root, dirs, files in os.walk(GH_path):
                    root_path = Path(root)
                    relative_path = root_path.relative_to(GH_path)
                    # For each file in the current directory
                    for file in files:
                        # Source file path
                        source_path = root_path / file
                        # Destination path includes the relative_path to preserve the directory structure
                        destination_path = lib_path / relative_path / file
                        # Ensure the destination directory exists
                        destination_path.parent.mkdir(parents=True, exist_ok=True)
                        # Copy the file to the destination
                        shutil.copy(source_path, destination_path)

        def reinstate_env(self, override=False):
            if input("Are you sure you want to reinstate the environment? \n"
                     "This will replace existing files in GH Libraries Folder"
                     "with those belonging to this environment? [y/n]").lower() == "y" \
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

        def clone_env(self, clone=False):
            GH_path = Path(self.gh_path)  # Ensure GH_path is a Path object
            lib_path = self.dirs["ghlib"]

            if clone:
                # os.walk() iterates over the directories and files in GH_path
                for root, dirs, files in os.walk(GH_path):
                    # Convert root to a Path object for easier manipulation
                    root_path = Path(root)

                    # Calculate the relative path from GH_path to the current directory
                    relative_path = root_path.relative_to(GH_path)

                    # For each file in the current directory
                    for file in files:
                        # Source file path
                        source_path = root_path / file

                        # Destination path includes the relative_path to preserve the directory structure
                        destination_path = lib_path / relative_path / file

                        # Ensure the destination directory exists
                        destination_path.parent.mkdir(parents=True, exist_ok=True)

                        # Copy the file to the destination
                        shutil.copy(source_path, destination_path)

        def reinstate_env(self, override=False):
            if input("Are you sure you want to reinstate the environment? \n"
                     "This will replace existing files in GH Libraries Folder"
                     "with those belonging to this environment? [y/n]").lower() == "y" \
                    or override:
                print("Reinstating environment...")
                GH_path = self.gh_path
                lib_path = self.dirs["ghlib"]

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
            for root, dirs, files in os.walk(self.dirs["ghlib"]):
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

        def initialise(self, vanilla_components: Path = None, clone_env=True):
            print("Setting environment variables")
            print("Copying vanilla components")
            if vanilla_components is None:
                vanilla_components = Path(EnvironmentManager.VANILLA_PATH)
            if vanilla_components:
                # Ensure the destination directory exists
                vanilla_components_destination = self.dirs["vanilla"]
                vanilla_components_destination.mkdir(parents=True, exist_ok=True)
                # Construct the full destination path including the filename
                destination_path = vanilla_components_destination / vanilla_components.name
                # Use the adjusted destination path that includes the filename
                self.copy_file(vanilla_components, destination_path)
            print("Copying components")
            self.clone_env(clone_env)
            print("Copying gh files")
            self.copy_ghtest_files()


        def copy_ghtest_files(self):
            for root, dirs, files in os.walk(EnvironmentManager.GH_FILEPATH):
                for file in files:
                    if file.endswith(".gh"):
                        shutil.copy(os.path.join(root, file), self.dirs["files"])
        def get_test_files(self):
            for root, dirs, files in os.walk(self.dirs["files"]):
                for file in files:
                    if file.endswith(".gh"):
                        yield os.path.join(root, file)
            print("No more gh files")


    @classmethod
    def load_environment(cls, environment_name):
        if environment_name not in cls._environments:
            cls._environments[environment_name] = cls.Environment(environment_name)
        return cls._environments[environment_name]

    @classmethod
    def get_directory_names(cls):
        # Returns the directory names defined at the class level
        return list(cls.DIR_STRUCTURE.values())



