import datetime
import pickle
import shutil

import rhinoinside
rhinoinside.load()
import clr
import sys
import System
import csv
import os
import matplotlib.colors as mcolors
import configparser

import torch

import matplotlib
path = r"C:\Program Files\Rhino 8\Plug-ins\Grasshopper"
sys.path.append(path)
clr.AddReference("Grasshopper")
clr.AddReference("GH_IO")
clr.AddReference("GH_Util")

import Grasshopper
import Grasshopper.GUI
import Grasshopper.Kernel as ghk
from Grasshopper.Kernel import GH_Document
from Grasshopper.Kernel import IGH_Component
from Grasshopper.Kernel import IGH_Param
from Grasshopper.Kernel import GH_DocumentIO
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch
from pathlib import Path
import re
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                    filemode="w",
                    filename="tests")
# create the loggers

import logging


def create_named_logger(name, filename):
    # Create a logger with the specified name
    logger = logging.getLogger(name)

    # Set the log level to INFO
    logger.setLevel(logging.DEBUG)

    # Create a file handler
    handler = logging.FileHandler(filename, mode='w')

    # Set the level of the handler to INFO
    handler.setLevel(logging.DEBUG)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add the formatter to the handler
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger


logging_location = r"C:\Users\jossi\Dropbox\Office_Work\Jos\GH_Graph_Learning\logs"


# Usage

def debug_obj(obj):
    complog.debug("The object is not an IGH_Component or IGH_Param.")
    complog.debug(f"Name: {obj.Name}")
    complog.debug(f"Nickname: {obj.NickName}")
    complog.debug(f"Guid: {obj.ComponentGuid}")
    complog.debug(f"IGHComponent: {GHComponent.is_cls(obj, ghk.IGH_Component)}")
    complog.debug(f"IGHParam: {GHComponent.is_cls(obj, ghk.IGH_Param)}")


complog = create_named_logger('components', Path(logging_location) / 'complog.log')
complog.info('Loggging started')
filelogger = create_named_logger("file_logger", Path(logging_location) / "file_logger.log")
filelogger.info('Loggging started')

class EnvironmentManager:
    _environments = {}  # Stores Environment instances keyed by environment_name

    def __init__(self, config_file):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

        self.DIR_STRUCTURE = self.config._sections['DIR_STRUCTURE']
        self.GHPATH = self.config.get('PATHS', 'GHPATH')
        self.VANILLA_PATH = self.config.get('PATHS', 'VANILLA_PATH')
        self.GH_FILE_PATH = self.config.get('PATHS', 'GH_FILE_PATH')

    class Environment:
        def __init__(self, environment_name, config):
            self.environment_name = environment_name
            self.base_path = Path(config.get('PATHS', 'BASE_PATH')) / f"{environment_name}"
            self.base_path.mkdir(parents=True, exist_ok=True)
            
            self.dirs = {name: self.base_path / dirname for name, dirname in config._sections['DIR_STRUCTURE'].items()}
            new_dirs = {}
            for path in list(self.dirs.values()):
                path.mkdir(parents=True, exist_ok=True)
                if path.name == "03-GH_Files":
                    raw = "03a-Raw"
                    raw_path = path / raw
                    raw_path.mkdir(parents=True, exist_ok=True)
                    new_dirs["raw"] = raw_path
                    processed = "03b-Processed"
                    processed_path = path / processed
                    processed_path.mkdir(parents=True, exist_ok=True)
                    new_dirs["processed"] = processed_path

            self.dirs.update(new_dirs)

            self.gh_path = Path(config.get('PATHS', 'GHPATH'))
            self.initialise(config)

        def serialize(self):
            file_path = self.base_path / "environment.pkl"
            with open(file_path, "wb") as file:
                pickle.dump(self, file)

        def clone_env(self, clone=False):
            GH_path = Path(self.gh_path)
            lib_path = self.dirs.get('ghlib')
            r, d, f = next(os.walk(lib_path))
            if clone and len(f) == 0:
                for root, dirs, files in os.walk(GH_path):
                    root_path = Path(root)
                    relative_path = root_path.relative_to(GH_path)
                    for file in files:
                        source_path = root_path / file
                        destination_path = lib_path / relative_path / file
                        destination_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(source_path, destination_path)

        def reinstate_env(self, override=False):
            if input("Are you sure you want to reinstate the environment? \n"
                     "This will replace existing files in GH Libraries Folder"
                     "with those belonging to this environment? [y/n]").lower() == "y" \
                    or override:
                print("Reinstating environment...")
                GH_path = self.gh_path
                lib_path = self.dirs["ghlib"]

                for root, dirs, files in os.walk(GH_path):
                    for file in files:
                        file_path = Path(root) / file
                        if file_path.is_file():
                            file_path.unlink()

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
            if not src.exists() or not src.is_file():
                print(f"Source file does not exist or is not a file: {src}")
                return

            dst.parent.mkdir(parents=True, exist_ok=True)

            try:
                shutil.copyfile(src, dst)
                print(f"File copied successfully from {src} to {dst}.")
            except Exception as e:
                print(f"Error copying file from {src} to {dst}: {e}")

        def initialise(self, config, vanilla_components: Path = None, clone_env=True):
            print("Setting environment variables")
            print("Copying vanilla components")
            if vanilla_components is None:
                vanilla_components = Path(config.get('PATHS', 'VANILLA_PATH'))
            if vanilla_components:
                vanilla_components_destination = self.dirs["vanilla"]
                vanilla_components_destination.mkdir(parents=True, exist_ok=True)
                destination_path = vanilla_components_destination / vanilla_components.name
                self.copy_file(vanilla_components, destination_path)
            print("Copying components")
            self.clone_env(clone_env)
            print("Copying gh files")
            self.copy_ghtest_files(config)

        def copy_ghtest_files(self, config):
            try:
                for root, dirs, files in os.walk(config.get('PATHS', 'GH_FILE_PATH')):
                    for file in files:
                        if file.endswith(".gh"):
                            shutil.copy(os.path.join(root, file), self.dirs["raw"])
            except PermissionError as e:
                print(f"Permission error on {e}")

        def get_gh_file(self):
            for root, dirs, files in os.walk(self.dirs["files"]):
                for file in files:
                    if file.endswith(".gh"):
                        yield os.path.join(root, file)
            print("No more gh files")

    def create_environment(self, environment_name):
        if environment_name not in self._environments:
            self._environments[environment_name] = self.Environment(environment_name, self.config)
        return self._environments[environment_name]

    def get_directory_names(self):
        return list(self.DIR_STRUCTURE.values())

    def get_environment(self):
        return list(self._environments.values())[0]

    def setup_logging(self, environment_name):
        log_directory = self._environments[environment_name].dirs['logs']
        log_file_path = log_directory / "GH.log"
        print(log_file_path)
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                            filemode="w",
                            filename=str(log_file_path))

class GHComponentProxy:
    """
    A class that represents an uninstantiated node component. This class is used to
    interface with the Grasshopper environment and manage component metadata.
    """
    component_server = Grasshopper.Instances.ComponentServer

    def __init__(self, obj_proxy: ghk.IGH_ObjectProxy):
        self.obj_proxy = obj_proxy
        self.category = str(obj_proxy.Desc.Category)
        self.name = str(obj_proxy.Desc.Name)
        self.guid = str(obj_proxy.Guid)
        self.nickname = str(obj_proxy.Desc.NickName)
        self.description = str(obj_proxy.Desc.Description)
        self.type = str(obj_proxy.Type)
        self.library = self.get_assembly_name()
        self.inputs, self.outputs = self.process_parameters()

    @property
    def sys_guid(self):
        return System.Guid(self.guid)

    @property
    def lib_guid(self):
        return self.obj_proxy.LibraryGuid

    def to_dict(self):
        return {
            "category": self.category,
            "name": self.name,
            "guid": self.guid,
            "nickname": self.nickname,
            "description": self.description[:60].replace("\n", " ").replace("\r", " ").strip(),
            "type": self.type,
            "library": self.library,
            'paramin': self.inputs,
            'paramout':self.outputs
        }

    def get_assembly(self) -> System.Reflection.Assembly:
        """
        Returns the AssemblyInfo object of a component. This method interfaces with
        the Grasshopper component server to retrieve assembly information.
        """
        return GHComponentProxy.component_server.FindAssembly(self.lib_guid)

    def get_assembly_name(self):
        """
        Retrieves the name of the assembly containing the component. This is used to
        identify the source library of the component.
        """
        assembly = self.get_assembly()
        return assembly.Name if assembly else "Unknown"

    def process_parameters(self):
        comp_data = {'inputs': [], 'outputs': []}
        prox = self.obj_proxy
        name = prox.Desc.Name
        p = prox.CreateInstance()
        try:
            p = IGH_Component(p)
            params = [param for param in p.Params.GetEnumerator()]
            kinds = [int(prm.Kind) for prm in params]
            typnames = [prm.TypeName for prm in params]
            param_properties = list(zip(typnames, kinds))
            for t, n in param_properties:
                if n == 2:
                    comp_data['inputs'].append(t)
                elif n == 3:
                    comp_data['outputs'].append(t)
        except Exception as e:
            try:
                p = IGH_Param(p)
                k = p.TypeName
                comp_data['inputs'].append(k) # Assuming this was intended to be managed differently.
                comp_data['outputs'].append(k) 
            except Exception as e:
                pass
        return comp_data["inputs"], comp_data["outputs"]

    def __str__(self):
        return f"{self.category}:{self.name}"

    def __repr__(self):
        return self.__str__()


class GHComponentTable:
    cs = Grasshopper.Instances.ComponentServer.ObjectProxies
    compserver = Grasshopper.Instances.ComponentServer
    component_dict = {component.Guid: component for component in cs}
    vanilla_proxies = {}
    object_proxies = []
    non_native_proxies = []
    _guid_to_idx = {}
    _idx_to_guid = {}
    df = None

    @classmethod
    def initialise(cls, environment_manager, vanilla_components_location=None):
        if vanilla_components_location is None:
            vanilla_components_location = r"C:\Users\jossi\Dropbox\Office_Work\Jos\GH_Graph_Learning\Grasshopper Components\240318-VanillaComponents"
        cls.vanilla_proxies = {obj.sys_guid: obj for obj in cls.load_vanilla_gh_proxies(vanilla_components_location)}
        cls.object_proxies = [GHComponentProxy(obj) for obj in cls.cs]
        cls.non_native_proxies = sorted(
            [ghp for ghp in cls.object_proxies if not cls.is_native(ghp)],
            key=lambda x: str(x))
        cls.object_proxies = list(cls.vanilla_proxies.values()) + cls.non_native_proxies
        env = environment_manager
        df_directory = env.dirs['all']
        cls.obj_lookup = {obj.sys_guid: obj for obj in cls.object_proxies}
        cls.df = cls.table_to_pandas(df_directory, 'all_components')
        cls.to_csv()

    @classmethod
    def to_csv(cls, location=None, name="grasshopper_components.csv"):
        if location is None:
            location = r"C:\Users\jossi\Dropbox\Office_Work\Jos\GH_Graph_Learning\Grasshopper Components\240318-AllComponents"
        filename = Path(location) / name
        if not filename.exists():
            with open(filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                if cls.object_proxies:
                    header = cls.object_proxies[0].to_dict().keys()
                    writer.writerow(header)
                    for proxy in cls.object_proxies:
                        writer.writerow(proxy.to_dict().values())


    @classmethod
    def export_vanilla_gh_table(cls, filepath):
        datetime_str = datetime.datetime.now().strftime("%y%m%d")
        filename = Path(filepath) / str(str(datetime_str) + " vanilla_components.csv")
        if not filename.exists():
            with open(filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                if cls.object_proxies:
                    header = cls.object_proxies[0].to_dict().keys()
                    writer.writerow(header)
                    for proxy in cls.object_proxies:
                        writer.writerow(proxy.to_dict().values())
                        print(f"Exported vanilla components to CSV: {filename}")
                else:
                    cls.object_proxies = sorted([GHComponentProxy(obj) for obj in cls.cs], key=lambda x: x.type)
                    with open(filename, mode='w', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)
                        if cls.object_proxies:
                            header = cls.object_proxies[0].to_dict().keys()
                            writer.writerow(header)
                            for proxy in cls.object_proxies:
                                writer.writerow(proxy.to_dict().values())
                            print(f"Exported vanilla components to CSV: {filename}")


    @classmethod
    def search_component_by_guid(cls, guid: System.Guid):
        return cls.component_dict[guid]

    @classmethod
    def is_native(cls, ghp):
        return cls.vanilla_proxies.get(ghp.sys_guid, False)

    @classmethod
    def convert_csv_line_to_proxies(cls, guid_str: str):
        guid = System.Guid(guid_str)
        proxy_object = cls.search_component_by_guid(guid)
        if proxy_object:
            logging.debug(f"Proxy object: {proxy_object.Desc.Name} found")
            return GHComponentProxy(proxy_object)
        logging.warning(f"Could not find a vanilla GH with guid {guid_str}")
        return None

    @classmethod
    def load_vanilla_gh_proxies(cls, filepath, file='vanilla_components.csv'):
        vanilla_proxies = []
        location = Path(filepath) / file
        if location.exists():
            with open(location, mode='r') as cc:
                reader = csv.DictReader(cc)
                for row in reader:
                    guid_str = row['guid']
                    gh_proxy = cls.convert_csv_line_to_proxies(guid_str)
                    if gh_proxy:
                        vanilla_proxies.append(gh_proxy)
            return vanilla_proxies
        logging.warning(f"Could not find a vanilla components file: {filepath}")

    @classmethod
    def view_all_categories(cls):
        categories = set([pr.category for pr in cls.object_proxies])
        return categories

    @classmethod
    def view_proxies(cls, n):
        print(f"There are {len(cls.object_proxies)} proxies")
        for proxy in cls.object_proxies[:n]:
            print(proxy)

    @classmethod
    def table_to_pandas(cls, location, filename='all_components.csv'):
        full_path = Path(location) / filename

        # Ensure the CSV exists, if not, create it
        if not full_path.exists():
            cls.to_csv(location, filename)  # Make sure to pass both location and filename

        # Now, the file should exist. Read it into a pandas DataFrame
        df = pd.read_csv(full_path)  # Use full_path, which is the complete path to the file

        # Building the GUID to index mappings
        cls._guid_to_idx = {row['guid']: idx for idx, row in df.iterrows()}
        cls._idx_to_guid = {idx: row['guid'] for idx, row in df.iterrows()}
        return df


    @classmethod
    def guid_to_component_name(cls, guid, category=True):
        if type(guid) != System.Guid:
            guid = System.Guid(guid)
        proxy = cls.component_dict.get(guid)
        if category:
            return f"{proxy.Desc.Category}:{proxy.Desc.Name}"
        return f":{proxy.Desc.Name}"
    @classmethod
    def get_guid_to_idx(cls, guid: System.Guid) -> int:
        return cls._guid_to_idx.get(str(guid), None)

    @classmethod
    def get_idx_to_guid(cls, idx: int) -> System.Guid:
        guid_str = cls._idx_to_guid.get(idx, None)
        return System.Guid(guid_str) if guid_str else None

    @classmethod
    def component_to_idx(cls, component) -> int:
        component_category = component.category

        component_name = component.name
        if cls.df is not None:
            id = str(component.obj.ComponentGuid)
            q = cls.df.query("`guid` == @id")
            if q.empty:
                raise ValueError(f"Component {component_name} not found in dataframe.")
            else:
                return q.index[0]
        else:
            raise ValueError(f"Cant Search an empty dataframe.")

    @classmethod
    def idx_to_component(cls, idx):
        return cls.df["nickname"].iloc[idx]

    @classmethod
    def idx_to_component2(cls, idx):
        return cls.df["name"].iloc[idx]


    @classmethod
    def get_guid_by_type(cls, type_search):
        # this is used when preprocessing a file and returns the component id given the object type
        matches = cls.df[cls.df['type'] == type_search]
        if not matches.empty:
            return matches['guid'].tolist()
        else:
            return []

class GHDocumentPreprocessor:
    def __init__(self, folder_path: str):
        self.folder_path = Path(folder_path)

    def process_folder_or_file(self, illegals_dict ,overwrite: bool = True):
        if self.folder_path.is_file() and self.folder_path.suffix == ".gh":
            # Process a single file
            doc = GHProcessor.get_ghdoc(str(self.folder_path))
            if doc:
                doc = self.preprocess_and_replace(doc= doc,
                                                  file=self.folder_path,
                                                  illegals_dict=illegals_dict,
                                                  overwrite=overwrite)
                return doc
            else:
                print(f"Failed to open file: {self.folder_path}")
        elif self.folder_path.is_dir():
            # Process all files in the folder
            docs = []
            for file in self.folder_path.glob("*.gh"):
                doc = GHProcessor.get_ghdoc(str(file))
                if doc:
                    doc = self.preprocess_and_replace(doc= doc,
                                                  file=file,
                                                  illegals_dict=illegals_dict,
                                                  overwrite=overwrite)
                    docs.append(doc)
                else:
                    print(f"Failed to open file: {file}")
            return docs
        else:
            print(f"Invalid path: {self.folder_path}")

    def remove_unwanted_items(self, doc, illegals_dict: Dict):
        illegals_set = set([str(v) for v in illegals_dict.values()])
        removables = [obj for obj in doc.Objects if str(obj.ComponentGuid) in illegals_set]
        for obj in removables:
            doc.RemoveObject(obj, True)
        return doc

    def doc_save(self, doc, location):
        ghdio = GH_DocumentIO(doc)
        ghdio.SaveQuiet(str(location))


    def get_component_type(self, component):
        pattern = r"^(.*?)_OBSOLETE"
        match = re.search(pattern, str(component))
        return match.group(1) if match else None

    def create_new_component(self, component_type: str):
        replacement_guids = GHComponentTable.get_guid_by_type(component_type)
        if replacement_guids:
            new_component_proxy = GHComponentTable.search_component_by_guid(System.Guid(replacement_guids[0]))
            if new_component_proxy:
                return new_component_proxy.CreateInstance()
        return None

    def rewire_connections(self, old_component, new_component):
        old_component_obj = GHComponent.convert_cls(old_component, ghk.IGH_Component)
        new_component_obj = GHComponent.convert_cls(new_component, ghk.IGH_Component)

        if old_component_obj and new_component_obj:
            for input, new_input in zip(old_component_obj.Params.Input, new_component_obj.Params.Input):
                for source in input.Sources:
                    new_input.AddSource(source)
            for output, new_output in zip(old_component_obj.Params.Output, new_component_obj.Params.Output):
                for recipient in output.Recipients:
                    recipient.AddSource(new_output)

        else:
            old_param_obj = GHComponent.convert_cls(old_component, ghk.IGH_Param)
            new_param_obj = GHComponent.convert_cls(new_component, ghk.IGH_Param)

            if old_param_obj and new_param_obj:
                for source in old_param_obj.Sources:
                    new_param_obj.AddSource(source)
                for recipient in old_param_obj.Recipients:
                    recipient.AddSource(new_param_obj)

            else:
                print(f"Cannot rewire connections. Incompatible component types: {type(old_component)} and {type(new_component)}")

    def replace_component(self, doc, old_component, new_component):
        if new_component:
            new_component.Attributes.Pivot = old_component.Attributes.Pivot
            doc.AddObject(new_component, False)
            self.rewire_connections(old_component, new_component)
            doc.RemoveObject(old_component, True)
            return True
        print(f"no new component, did not replace {old_component.Name}")
        return False

    def manual_replace_obsolete_components(self, obj, doc):

        component = obj
        component_type = self.get_component_type(obj)
        if component_type:
            print(f"old component: {component_type}")
            new_component = self.create_new_component(component_type)
            print(f"new component: {new_component}")
            if new_component:
                if not self.replace_component(doc, component, new_component):
                    print(f"Could not replace obsolete component: {component.Name}")
        return doc

    def replace_obsolete_components(self, doc):
        components = [obj for obj in doc.Objects]
        for obj in components:
            if obj.Obsolete:
                upgrader = GHComponentTable.compserver.FindUpgrader(obj.ComponentGuid)
                if upgrader:
                    upgrader.Upgrade(obj, doc)
                    print(f"Upgraded {obj.Name}")
                else:
                    temp_doc = self.manual_replace_obsolete_components(obj,doc)
                    if temp_doc:
                        doc = temp_doc
                    else:
                        print(f"No upgrader found for {obj.Name}")
        print("Replacement of obsolete components complete.")
        return doc

    def remove_placeholder_components(self, doc, update=False):
        placeholders = []
        for component in doc.Objects:
            a = GHComponentTable.get_guid_to_idx(str(component.ComponentGuid))
            if a is None:
                placeholders.append(component)
        if update:
            ghk.GH_Document.NewSolution(doc, False)

        for placeholder in placeholders:
            doc.RemoveObject(placeholder, False)
        return doc

    def move_file_to_error_bin(self, file: Path, error_message: str = ""):
        errorbin = EnvironmentManager.get_environment().dirs['processed']/ "error_bin"
        if not errorbin.exists():
            errorbin.mkdir()
        try:
            if not file.exists():
                print(f"File does not exist: {file}")
                return
            new_file_path = errorbin / file.name
            shutil.move(str(file), str(new_file_path))
            if error_message:
                print(error_message)
            print(f"File moved to error bin: {new_file_path}")
        except Exception as e:
            print(f"Failed to move file to error bin: {e}")

    def preprocess_and_replace(self, doc, file, illegals_dict, overwrite: bool = True):
        doc = self.remove_unwanted_items(doc=doc, illegals_dict=illegals_dict)
        doc = self.replace_obsolete_components(doc)
        doc = self.remove_placeholder_components(doc)
        if overwrite:
            try:
                self.doc_save(doc, location=EnvironmentManager.get_environment().dirs['processed']/file.name)
            except Exception as e:
                print(f"Failed to save file: {e}, moved {file} to error bin.")
                self.move_file_to_error_bin(file, e)

        print("Preprocessing, replacement, and placeholder removal complete.")
        return doc


class GHNode:
    def __init__(self, obj: ghk.IGH_DocumentObject):
        self.obj = obj
        self.category = obj.Category
        self.name = obj.Name
        self.id = str(obj.InstanceGuid)
        self.pos = obj.Attributes.Pivot if hasattr(obj.Attributes, "Pivot") else None
        self.X = None
        self.Y = None
        if self.pos:
            self.X = self.pos.X
            self.Y = self.pos.Y
        self.uid = str(obj.ComponentGuid)
        # Assuming global_idx is somehow related to GHComponentTable, which might need instance reference
        self.global_idx = GHComponentTable.component_to_idx(self)  # This requires GHComponentTable method adjustment
        self.graph_id = None

    def get_recipients(self):
        """To be implemented by the subclass"""
        pass

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return f"<GHNode {self.__str__()}>"

    def log_properties(self):
        log = {
            f"Category: {self.category}, "
            f"Name: {self.name}, "
            f"ID: {self.id[-5:]}, "
            f"Position: {self.pos}"
            f"Global: {self.global_idx}"
        }
        # This method seems intended for logging or debugging, consider how it's used and adapt accordingly.


class GHParam:

    def __init__(self, obj):
        self.obj = obj
        self.parent = GHNode(ghk.IGH_DocumentObject(obj).Attributes.GetTopLevel.DocObject)
        self.name = obj.Name
        self.datamapping = int(obj.DataMapping)  # enumerator 0:none, 1:Flatten, 2:Graft
        self.access = int(obj.Access)  # the access: item, list, tree
        self.pkind = obj.Kind  # the kind: floating (top level), input (parameter tied to component as input), output (parameter tied to a component as an output
        self.dataEmitter = obj.IsDataProvider  # boolean stating whether this object is able to emit data
        self.optional = int(obj.Optional)  # gets whether this parameter is optional to the functioning of the component
        self.typname = obj.TypeName  # human-readable descriptor of this parameter
        self.optional = obj.Optional  # gets whether this parameter is optional to the functioning of the component
        # logging.info(f'GHComponent {self.parent.name} Params: {self.log_properties()}')

    @property
    def recipients(self):
        # if there are no recipents to this parameter, return none
        return [rcp for rcp in self.obj.Recipients] if len(self.obj.Recipients) > 0 else None

    @property
    def sources(self):
        # if there are no recipents to this parameter, return none
        return [rcp for rcp in self.obj.Sources] if len(self.obj.Sources) > 0 else None

    @property
    def data(self):
        return self.obj.VolatileData.DataDescription(False, False)

    def log_properties(self):
        properties = (
            f"Name: {self.name}, "
            f"DataMapping: {self.datamapping}, "
            f"Kind: {self.pkind}, "
            f"DataEmitter: {self.dataEmitter}, "
            f"TypeName: {self.typname}, "
            f"Optional: {self.optional}, "
            f"Data: {self.data}, "
        )
        return properties

    def __str__(self):
        repr_obj = ghk.IGH_DocumentObject(self.obj)
        return f"param:{self.name}"

    def __repr__(self):
        return f"<GHParam {self.__str__()}>"


class GHComponent(GHNode):
    """Subclass of GHNode that handles GH components that implement IGH_Component.
    Each GHComponent object should contain a list of input parameter and output parameter objects.
    These parameter objects have access to the sources and recipients of the parameter"""

    def __init__(self, obj):
        super().__init__(obj)
        self.obj: Union[IGH_Component or IGH_Param] = None
        self.iparams = []
        self.oparams = []
        self.recipients = []
        self.iparams_dict = {}
        self.oparams_dict = {}

        # Attempt to initialize parameters
        self.initialize_parameters(obj)
        self.initialize_lookup_dicts()

    def initialize_parameters(self, obj):
        if GHComponent.is_cls(obj, ghk.IGH_Component):
            self.obj = IGH_Component(obj)
            self.iparams = [GHParam(p) for p in self.obj.Params.Input]
            self.oparams = [GHParam(p) for p in self.obj.Params.Output]

        elif GHComponent.is_cls(obj, ghk.IGH_Param):
            self.obj = IGH_Param(obj)
            param = GHParam(self.obj)
            self.iparams = [param]
            self.oparams = [param]
        else:
            debug_obj(obj)
            raise TypeError("The object is not an IGH_Component or IGH_Param.")

    def initialize_lookup_dicts(self):
        self.iparams_dict = {k.name: i for i, k in enumerate(self.iparams)}
        complog.debug(f"{self.name}: iparams dict: {self.iparams_dict}")
        self.oparams_dict = {k.name: i for i, k in enumerate(self.oparams)}
        complog.debug(f"{self.name}: iparams dict: {self.oparams_dict}")

    @staticmethod
    def is_cls(obj, clas):
        try:
            clas(obj)
            return True
        except Exception:
            return False
    @staticmethod
    def convert_cls(obj, clas):
        try:
            return clas(obj)
        except Exception:
            return None

    def get_connections(self, canvas):
        """Returns the source and recipient connections for this component"""
        source_connections = []
        recipient_connections = []

        # Handle connections to recipients from this component's output parameters
        if self.oparams is not None:
            for i, oparam in enumerate(self.oparams):  # Iterate over output parameters
                if oparam.recipients:  # Ensure there are recipients to consider
                    this_access = oparam.access
                    this_datamapping = oparam.datamapping
                    this_optional = oparam.optional
                    source_datamapping = {'instanceid': self.id,
                                          'compid': self.uid,
                                          "paramidx": i,
                                          'access': this_access,
                                          'dm': this_datamapping,
                                          'optional': this_optional}
                    for r in oparam.recipients:
                        recipient = GHParam(r)
                        recipient_access = recipient.access
                        recipient_datamapping = recipient.datamapping
                        recipient_optional = recipient.optional

                        # Search the canvas for the corresponding objects. Remember to convert the InstanceGUID into a str
                        recipient_component = canvas.find_object_by_guid(
                            str(recipient.parent.obj.Attributes.InstanceGuid))
                        if recipient_component is not None:
                            parent_instance = canvas.find_object_by_guid(recipient_component.id)
                            if parent_instance is not None:
                                recipient_parameter_index = recipient_component.iparams_dict.get(recipient.name)
                                if recipient_parameter_index is not None:
                                    recipient_datamapping = {'instanceid': parent_instance.id,
                                                             'compid': recipient_component.uid,
                                                             'paramidx': recipient_parameter_index,
                                                             'access': recipient_access,
                                                             'dm': recipient_datamapping,
                                                             'optional': recipient_optional}
                                    connection = {"source": source_datamapping, "recipient": recipient_datamapping}
                                    recipient_connections.append(connection)

        if self.iparams is not None:
            # Handle connections from sources to this component's input parameters
            for i, iparam in enumerate(self.iparams):  # Iterate over input parameters
                if iparam.sources:  # Ensure there are sources to consider
                    this_access = iparam.access
                    this_datamapping = iparam.datamapping
                    this_optional = iparam.optional
                    recipient_datamapping = {'instanceid': self.id,
                                             'compid': self.uid,
                                             'paramidx': i,
                                             'access': this_access,
                                             'dm': this_datamapping,
                                             'optional': this_optional}
                    for s in iparam.sources:
                        source = GHParam(s)
                        source_access = source.access
                        source_datamapping = source.datamapping
                        source_optional = source.optional

                        # Search the canvas for the corresponding objects. Remember to convert the InstanceGUID into a str
                        source_component = canvas.find_object_by_guid(str(source.parent.obj.Attributes.InstanceGuid))
                        if source_component is not None:
                            source_instance = canvas.find_object_by_guid(source_component.id)
                            if source_instance is not None:
                                source_parameter_index = source_component.oparams_dict.get(source.name)
                                if source_parameter_index is not None:
                                    source_datamapping = {'instanceid': source_instance.id,
                                                          'compid': source_component.uid,
                                                          'paramidx': source_parameter_index,
                                                          'access': source_access,
                                                          'dm': source_datamapping,
                                                          'optional': source_optional}
                                    connection = {"source": source_datamapping, "recipient": recipient_datamapping}
                                    source_connections.append(connection)

        return source_connections, recipient_connections

    def print_conns(self, canvas):
        source_conns, recipient_conns = self.get_connections(canvas)
        print(f"Component: {self.name}")
        print(f"Sources: {source_conns}")
        print(f"Recipients: {recipient_conns}")

    def __str__(self):
        return f"Comp:{self.name}"

    def __repr__(self):
        return f"<GHNComponent {self.__str__()}>"


class Canvas:
    def __init__(self, doc, environment, n=None):
        self.doc = doc
        self.components = []  # Initialize as empty
        self.guid_to_component = {}  # Initialize as empty
        self.initialize_components(n)
        self.env = environment

    def initialize_components(self, n=None):
        ignore = ["Group", "Sketch", "Scribble", "Cluster", "Gradient"]
        self.components = []
        try:
            for o in self.doc.Objects:
                if o.Name not in ignore:
                    try:
                        obj = GHComponent(o)
                        self.components.append(obj)
                    except TypeError:
                        logging.warning(f"Incompatible Component: {o.Name}")
                        print(f"Incompatible Component: {o.Name}")
                self.guid_to_component = {c.id: c for c in self.components}
        except TypeError:
            print("Can't initialise")
            logging.warning(f"Can't initialise canvas because a component is creating an error")

    def find_object_by_guid(self, guid: str) -> GHComponent:
        return self.guid_to_component.get(guid)
    def __str__(self):
        return str(self.components)

    def __repr__(self):
        return f"<Canvas {self.__str__()}>"


class GraphConnection:

    def __init__(self, x1, y1, v1n, v1i, x2, y2, v2n, v2i, edge):
        self.x1 = x1
        self.y1 = y1
        self.v1n = v1n
        self.v1i = v1i
        self.x2 = x2
        self.y2 = y2
        self.v2n = v2n
        self.v2i = v2i
        self.e1 =  edge[0]
        self.e2 = edge[1]
        self.tensor = torch.tensor( [x1, y1, v1n, v1i, edge[0], x2, y2, v2n, v2i, edge[1]],
                                    dtype=torch.int16)

    @property
    def edge(self):
        return (self.v1i, self.v1i), (self.v2n, self.v2i)

    @property
    def edgeproperties(self):
        return (self.e1, self.e2)

    def __str__(self):
        return f"GC({self.v1n}-{self.v1i}[{self.e1}], {self.v2n}-{self.v2i}[{self.e2}])"

    def __repr__(self):
        return f"GraphConn ({self.v1n}-{self.v1i}, {self.v2n}-{self.v2i})"


class GraphNode:
    def __init__(self, component:GHComponent, canvas: Canvas):
        self.graph_id: Tuple[int, int] = component.graph_id
        self.compid = component.uid
        self.component = component
        self.canvas: Canvas = canvas
        self.X = component.obj.Attributes.Pivot.X
        self.Y = component.obj.Attributes.Pivot.Y
        self.pos = (self.X, self.Y)

    def edges(self, bidirectional=False):
        """:Returns the edge tuple in the form (int, int), (int,int) where
        the tuple describes the graph node id. If bidirectional is true, both the sources and recipient
        edges are returned, otherwise only the recipient edges are returned"""

        connections = self.get_graph_connections()
        if bidirectional:
            return connections['sources'] + connections['recipients']
        else:
            return connections['recipients']

    def get_graph_connections(self):
        sources, recipients = self.component.get_connections(self.canvas)

        graph_connections = {
            "sources": [],
            "recipients": []
        }

        # Process recipient connections (outgoing)
        if recipients:
            complog.debug(f"Recipients: {recipients}")
            for connection in recipients:
                v2 = connection.get("to")

                v1n, v1i = self.graph_id
                x1= int(self.X)
                y1 = int(self.Y)
                v2n, v2i = v2.graph_id
                edge = connection.get("edge")
                x2 = int(v2.X)
                y2 = int(v2.Y)
                graph_connections["recipients"].append(GraphConnection(x1 ,y1, v1n, v1i, x2, y2,v2n, v2i, edge))

        # Process source connections (incoming)
        if sources:
            complog.debug(f"Sources: {sources}")
            for connection in sources:
                v1 = connection.get("from")
                v1n, v1i = v1.graph_id
                x1 = int(v1.X)
                y1 = int(v1.Y)
                v2n, v2i = self.graph_id  # For incoming, self.id is the destination
                x2= int(self.X)
                y2 = int(self.Y)
                edge = [int(x) for x in connection.get("edge")]
                graph_connections["sources"].append(GraphConnection(x1 ,y1, v1n, v1i, x2, y2,v2n, v2i, edge))

        return graph_connections
    @property
    def feature_vector(self):
        connections = self.edges()
        # Check if 'connections' is not empty
        if connections:
            # Verify each 'conn' in 'connections' has a '.tensor' attribute that is a tensor
            for conn in connections:
                if not hasattr(conn, 'tensor'):
                    print("Missing 'tensor' attribute.")
                elif not isinstance(conn.tensor, torch.Tensor):
                    print("The 'tensor' attribute is not a PyTorch Tensor.")
            connection_tensor = torch.stack([conn.tensor for conn in connections])
            return connection_tensor
        else:
            return torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int16)
        return None
    def __str__(self):
        return f"GraphNode: {self.component.name}"

    def __repr__(self):
        return f"<GraphNode {self.__str__()}>"


class GHGraph:
    def __init__(self, canvas):
        self.canvas = canvas
        self.components = self.canvas.components

    @property
    def nodes(self) -> List[GraphNode]:
        return [GraphNode(component, self.canvas) for component in self.components]

    def nxGraph(self, bidirectional=False) -> nx.Graph:
        gx = nx.DiGraph()
        # Step 1: Add all nodes to the graph
        for node in self.nodes:
            node_id = node.graph_id# Ensure this is a simple, hashable type

            gx.add_node(node_id, X=node.X, Y=node.Y, guid = node.compid)  # Explicitly add nodes

        # Step 2: Add edges to the graph
        for node in self.nodes:
            for edge in node.edges():
                if edge:
                    # Ensure the tuples are hashable and correspond to actual node identifiers
                    gx.add_edge((edge.v1n, edge.v1i), (edge.v2n, edge.v2i), connection = f"{edge.e1},{edge.e2}")

        return gx

    def graph_img(self, save_path, show=True):
        # Create a figure and axis object for a consistent plotting context
        fig, ax = plt.subplots(figsize=(20, 12))
        gx = self.nxGraph()

        # Generate category color map and assign colors
        category_color_map = self.generate_category_color_map()
        default_color = "grey"  # Default color as fallback
        node_colors = [category_color_map.get(self.canvas.graph_id_to_component.get(node).category,
                                              default_color) if self.canvas.graph_id_to_component.get(
            node) else default_color for node in gx.nodes]

        # Initialize default labels for all nodes
        default_label = "Unknown"  # Default label for nodes without specific data
        custom_labels = {node: (
            self.canvas.graph_id_to_component.get(node).name if self.canvas.graph_id_to_component.get(
                node) else default_label) for node in gx.nodes}

        # Choose a different layout to spread out nodes more
        pos = {node.graph_id: node.pos for node in self.nodes}  # Using custom positions from nodes

        # Draw the graph with node colors and custom labels on the created axis
        nx.draw(gx, pos, with_labels=True, labels=custom_labels, node_color=node_colors, node_size=1000,
                edge_color="gray", linewidths=0.5, font_size=10, ax=ax)

        # Create a legend for the categories and ensure it's properly displayed
        legend_handles = [Patch(color=color, label=category) for category, color in category_color_map.items()]
        ax.legend(handles=legend_handles, title='Categories', loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0)

        # Save the figure before showing it to avoid clearing it
        fig.savefig(str(save_path), bbox_inches="tight")

        if show:
            plt.show()




    def generate_category_color_map(self):
        categories = sorted(GHComponentTable.view_all_categories())
        # Adjust to use matplotlib.colormaps instead of plt.cm.get_cmap
        cmap = matplotlib.colormaps['tab20b']
        category_color_map = {}
        for i, category in enumerate(categories):  # Ensure categories are sorted for consistency
            color = cmap(i / len(categories))  # Normalize index to 0-1 range for color mapping
            category_color_map[category] = color
        return category_color_map

    def save_graph(self, location: Path):
        if location.suffix != '.graphml':
            location = location.with_suffix('.graphml')  # Correctly adds the .graphml extension
        gx = self.nxGraph()
        nx.write_graphml(gx, location.as_posix())  # Convert Path object to string if necessary


class GHProcessor:
    def __init__(self, filepath_filename, environment):
        self.doc = self.get_ghdoc(filepath_filename)
        self.canvas = Canvas(filepath_filename, self.doc, environment)
        self.GHgraph = GHGraph(self.canvas)
        self.filename = Path(filepath_filename).name

    @staticmethod
    def get_ghdoc(filepath_filename):
        ghfile = filepath_filename
        if not os.path.exists(ghfile):
            print("This file does not exists:", ghfile)
            raise FileNotFoundError(ghfile)
        ghdocIO = Grasshopper.Kernel.GH_DocumentIO()
        ghdocIO.Open(ghfile)
        ghdoc = ghdocIO.Document
        return ghdoc

    def show_components(self):
        for component in self.canvas.components:
            print(component)

    def build_graph(self):
        self.GHgraph = GHGraph(self.canvas)
        return self.GHgraph

    def save_graph(self, path):
        try:

            self.GHgraph.save_graph(path)
        except Exception as e:
            print(e)

    @staticmethod
    def preprocess(doc):
        illegals = {
            "bifocals": "aced9701-8be9-4860-bc37-7e22622afff4",
            "wombat-feature_request": "82fde8cd-3e18-4185-9ec4-e649cc993137"
        }
        for comp_name, comp_id in illegals.items():
            illegal = doc.FindComponent(comp_id)
            while illegal is not None:
                doc.RemoveObject(illegal)
                illegal = doc.FindComponent(comp_id)  # Check again to ensure all instances are removed
        print("preprocessing complete")


def create_and_save_graph(env: EnvironmentManager.Environment, doc: GH_Document, filename: Path = None,
                          show: bool = False):
    components = []
    # extract all the components from the document
    for obj in doc.Objects:
        try:
            components.append(GHComponent(obj))
        except TypeError:
            print(f"Error in {obj}")
            continue
    canvas = Canvas(components, doc, env)
    # Create a directed graph
    graph = nx.DiGraph()

    pos = {}  # Dictionary to store the positions of nodes
    cat = {}  # Dictionary to store the categories of nodes

    # get all the connections from the graph
    for comp in components:
        instance_id = comp.id
        component_id = comp.uid
        recipient_components = []
        connections_list = comp.get_connections(canvas)[1]

        pos[instance_id] = (comp.X, comp.Y)  # Store the position of the current node
        cat[instance_id] = comp.category  # Store the category of the current node

        graph.add_node(instance_id, x=comp.X, y=comp.Y, globidx=component_id, category=comp.category)

        if connections_list:
            for connection in connections_list:
                comp_obs = connection['to']
                edge = connection['edge']
                recipient_components.append({'instance_id': comp_obs.id,
                                             'component_id': comp_obs.uid,
                                             'conparam': edge})

                # Add the connected node to the graph if it doesn't exist
                if comp_obs.id not in graph.nodes:
                    graph.add_node(comp_obs.id, x=comp_obs.X, y=comp_obs.Y, globidx=comp_obs.uid,
                                   category=comp_obs.category)
                    pos[comp_obs.id] = (comp_obs.X, comp_obs.Y)  # Store the position of the connected node
                    cat[comp_obs.id] = comp_obs.category  # Store the category of the connected node

                # Add the edge to the graph, with the chosen features
                graph.add_edge(instance_id, comp_obs.id, paramfrom=edge[0], paramto=edge[1])

    # Create a figure
    fig = plt.figure(figsize=(10, 8))  # Adjust the width and height as needed

    # Get the unique categories
    categories = set(cat.values())

    # Create a color map based on the number of categories
    num_colors = len(categories)
    color_map = plt.cm.get_cmap('viridis', num_colors)

    # Create a dictionary mapping categories to colors
    category_colors = {}
    for i, category in enumerate(categories):
        category_colors[category] = mcolors.to_hex(color_map(i))

    # Create a list of colors based on the node categories
    node_colors = [category_colors[cat[node]] for node in graph.nodes()]

    # Draw the graph using the stored positions and colors
    nx.draw(graph, pos=pos, with_labels=False, font_weight='bold', node_size=10, node_color=node_colors,
            edge_color='gray', arrowsize=5)

    # Create a legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=category,
                                  markerfacecolor=color, markersize=8)
                       for category, color in category_colors.items()]
    plt.legend(handles=legend_elements, title='Categories', loc='upper right')
    plt.axis('equal')

    if show:
        plt.show()

    if filename:
        plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
        # Save the graph as a GraphML file
        nx.write_graphml(graph, f"{filename}.graphml")

    plt.close(fig)

def load_create_environment(environment_name, config):
    # Load or create the environment
    environment_manager = EnvironmentManager(config_file=config)
    env = environment_manager.create_environment(environment_name)
    # GHComponentTable.to_csv(env.dirs['all'])
    # Construct the path to the environment_registry.csv file
    registry_file = Path.cwd() / "environment_registry.csv"

    # Check if the environment already exists in the registry
    environment_exists = False
    if registry_file.exists():
        with open(registry_file, mode="r", encoding="utf-8") as file:
            for line in file:
                if environment_name in line:
                    environment_exists = True
                    break

    # If the environment doesn't exist, append it to the registry
    if not environment_exists:
        with open(registry_file, mode="a", encoding="utf-8") as file:
            # Get the current date
            today = datetime.date.today().strftime("%Y-%m-%d")
            # Construct the string to write to the file
            record = f"{today}, {environment_name}\n"
            # Write the constructed string to the file
            file.write(record)
    return env


def test_multiple(env, n):
    gh_files_generator = env.get_gh_file()  # Create the generator object
    i = 0
    for gh_file in gh_files_generator:
        try:
            print(f"Processing {gh_file}")
            if i >= n:  # Break the loop if the limit is reached
                break
            gh_file = str(gh_file)
            ghp = GHProcessor(gh_file, env)
            print(ghp.canvas.components)
        except Exception as e:
            logging.warning(f"Did not process file {gh_file}")
            print(f"#####___ERROR ON: {gh_file}")
            continue
        graph_name = Path(gh_file).stem  # More robust than splitting on "."
        ghp.GHgraph.show_graph(graph_name)  # Show graph does not return anything, so 'display' is not needed
        graph_path = env.dirs['graphml'] / f"{graph_name}"
        ghp.save_graph(graph_path)
        i += 1  # Increment the counter
        print(f"Saved graph: {graph_name}")