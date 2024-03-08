import datetime
import json
import os
import pickle
import shutil
from pathlib import Path
import logging
import json
import os
import networkx as nx
import rhinoinside
import logging
import nbimporter
from icecream import ic
rhinoinside.load()
import Rhino
import clr
import sys
import System
import io
import csv
import pandas as pd
from IPython.display import display
import torch
from typing import Dict, List
import networkx as nx
import matplotlib.pyplot as plt
path = r"C:\Program Files\Rhino 8\Plug-ins\Grasshopper"
sys.path.append(path)
clr.AddReference("Grasshopper")
clr.AddReference("GH_IO")
clr.AddReference("GH_Util")
import Grasshopper
import Grasshopper.Kernel as ghk
from Grasshopper.Kernel import IGH_Component
from Grasshopper.Kernel import IGH_Param
import traceback
from typing import Dict, List, Tuple
from pathlib import Path

import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                    filemode="w",
                    filename="GH.log")


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
            # Initialize directories based on the class-level DIR_STRUCTURE
            self.dirs = {name: self.base_path / dirname for name, dirname in EnvironmentManager.DIR_STRUCTURE.items()}
            for path in self.dirs.values():
                path.mkdir(parents=True, exist_ok=True)

            self.gh_path = Path(EnvironmentManager.GHPATH)
            self.initialise()


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
            try:
                for root, dirs, files in os.walk(EnvironmentManager.GH_FILE_PATH):
                    for file in files:
                        if file.endswith(".gh"):
                            shutil.copy(os.path.join(root, file), self.dirs["files"])
            except PermissionError as e:
                print(f"Permission error on {e}")


        def get_gh_file(self):
            for root, dirs, files in os.walk(self.dirs["files"]):
                for file in files:
                    if file.endswith(".gh"):
                        yield os.path.join(root, file)
            print("No more gh files")


    @classmethod
    def create_environment(cls, environment_name):
        if environment_name not in cls._environments:
            cls._environments[environment_name] = cls.Environment(environment_name)
        return cls._environments[environment_name]

    @classmethod
    def get_directory_names(cls):
        # Returns the directory names defined at the class level
        return list(cls.DIR_STRUCTURE.values())

    @classmethod
    def get_environment_names(cls):
        return list(cls._environments.keys())

    @classmethod
    def get_environment(cls):
        return list(cls._environments.values())[0]

    @classmethod
    def get_environment(cls, name):
        return cls._environments.get(name)


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
            "library": self.library
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

    def __str__(self):
        return f"{self.category}:{self.name}"

    def __repr__(self):
        return self.__str__()

class GHInstance:
    def __init__(self, vanilla_components_location, all_components_location):
        component_table = GHComponentTable
        component_table.initialise(vanilla_components_location)

class GHComponentTable:
    cs = Grasshopper.Instances.ComponentServer.ObjectProxies
    component_dict = {component.Guid: component for component in cs}
    vanilla_proxies = {}
    object_proxies = []
    non_native_proxies = []
    _guid_to_idx = {}
    _idx_to_guid = {}
    df = None

    @classmethod
    def initialise(cls, vanilla_components_location):
        cls.vanilla_proxies = {obj.sys_guid: obj for obj in cls.load_vanilla_gh_proxies(vanilla_components_location)}
        cls.object_proxies = [GHComponentProxy(obj) for obj in cls.cs]
        cls.non_native_proxies = sorted(
            [ghp for ghp in cls.object_proxies if not cls.is_native(ghp) and not ghp.obj_proxy.Obsolete],
            key=lambda x: str(x))
        cls.object_proxies = list(cls.vanilla_proxies.values()) + cls.non_native_proxies
        env = EnvironmentManager.get_environment()
        df_directory = env.dirs['all']
        cls.obj_lookup = {obj.sys_guid: obj for obj in cls.object_proxies}
        cls.df = cls.table_to_pandas(df_directory, 'all_components')


    @classmethod
    def to_csv(cls,location, name="grasshopper_components.csv"):
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
    def get_guid_to_idx(cls, guid: System.Guid) -> int:
        return cls._guid_to_idx.get(str(guid), None)

    @classmethod
    def get_idx_to_guid(cls, idx: int) -> System.Guid:
        guid_str = cls._idx_to_guid.get(idx, None)
        return System.Guid(guid_str) if guid_str else None

    @classmethod
    def component_to_idx(cls, component) -> int:
        component = ghk.IGH_DocumentObject(component.obj)
        guid = component.ComponentGuid
        component_name = component.Name
        component_category = component.Category
        idx = cls.get_guid_to_idx(guid)
        if idx is not None:
            return idx
        guid_str = cls.df.where(
            (cls.df['category'] == component_category) & (cls.df['name'] == component_name)).dropna().guid.iloc[0]
        if guid_str:
            idx = cls.get_guid_to_idx(System.Guid(guid_str))
            if idx:
                return idx
        logging.warning(
            f"ComponentTable.component_to_idx: Did not find the GUID or table match for {component_category}, {component_name}")
        return -1

    @classmethod
    def idx_to_component(cls, idx):
        return cls.df["nickname"].iloc[idx]

class GHNode:
    def __init__(self, obj: ghk.IGH_DocumentObject):
        self.obj = obj
        self.category = obj.Category
        self.name = obj.Name
        self.id = str(obj.InstanceGuid)
        self.position = obj.Attributes.Pivot if hasattr(obj.Attributes, "Pivot") else None
        self.uid = f"{self.category}_{self.name}_{self.id[-5:]}"
        # Assuming global_idx is somehow related to GHComponentTable, which might need instance reference
        self.global_idx = GHComponentTable.component_to_idx(self)  # This requires GHComponentTable method adjustment
        self.graph_id = None

    def get_recipients(self):
        """To be implemented by the subclass"""
        pass

    def __str__(self):
        return f"{self.uid}"

    def __repr__(self):
        return f"<GHNode {self.__str__()}>"

    def log_properties(self):
        log = {
            f"Category: {self.category}, "
            f"Name: {self.name}, "
            f"ID: {self.id[-5:]}, "
            f"Position: {self.position}"
            f"Global: {self.global_idx}"
        }
        # This method seems intended for logging or debugging, consider how it's used and adapt accordingly.

class GHParam:

    def __init__(self, obj):
        self.obj = obj
        self.parent = GHNode(ghk.IGH_DocumentObject(obj).Attributes.GetTopLevel.DocObject)
        self.name = obj.Name
        self.datamapping = int(obj.DataMapping)  # enumerator 0:none, 1:Flatten, 2:Graft
        self.pkind = obj.Kind  # the kind: floating (top level), input (parameter tied to component as input), output (parameter tied to a component as an output
        self.dataEmitter = obj.IsDataProvider  # boolean stating whether this object is able to emit data
        # self.typ = obj.Type
        self.typname = obj.TypeName  # human-readable descriptor of this parameter
        self.optional = obj.Optional  # gets whether this parameter is optional to the functioning of the component
        logging.info(f'GHComponent {self.parent.name} Params: {self.log_properties()}')

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
        try:
            self.iparams = None
            self.oparams = None
            self.recipients = None
            try:
                if self.category == "Params":
                    self.obj = IGH_Param(obj)
                    self.iparams = [GHParam(self.obj)]
                    self.oparams = [GHParam(self.obj)]
                else:
                    self.obj = IGH_Component(obj)
                    self.iparams = [GHParam(p) for p in self.obj.Params.Input]
                    self.oparams = [GHParam(p) for p in self.obj.Params.Output]
                self.recipients = [(param.name, [GHParam(p).parent for p in param.recipients]) for param in self.oparams if param.recipients is not None]
            except TypeError as e:
                logging.warning(f"COMPONENT {self.name},{self.id[-5:]}  failed initial assignment of parameters: {e}")
                try:
                    self.obj = IGH_Param(obj)
                    self.iparams = [GHParam(self.obj)]
                    self.oparams = [GHParam(self.obj)]
                    logging.info(f"COMPONENT {self.name},{self.id[-5:]} successfully assigned parameters: {self.iparams}, {self.oparams}")
                except TypeError as e:
                    traceback.print_exc()
                    print(e.with_traceback())
                    logging.warning(f"COMPONENT {self.name},{self.id[-5:]} DID NOT EXTRACT PARAMETERS, not added to components table")
                    print(f"DID NOT ADD {self.name}")
        except TypeError as e:
            print(e)
        self.iparams_dict = {k.name: i for i, k in enumerate(self.iparams)}
        self.oparams_dict = {k.name: i for i, k in enumerate(self.oparams)}

    def get_connections(self, canvas):
        """Returns the source and recipient connections for this component"""

        source_connections = []
        recipient_connections = []

        # Handle connections to recipients from this component's output parameters
        for i, oparam in enumerate(self.oparams):  # Iterate over output parameters
            if oparam.recipients:  # Ensure there are recipients to consider
                for r in oparam.recipients:
                    recipient = GHParam(r)
                    #Search the canvas for the corresponding objects. Remember to convert the InstanceGUID into a str
                    recipient_component = canvas.find_object_by_guid(str(recipient.parent.obj.Attributes.InstanceGuid))
                    parent_instance = canvas.find_object_by_guid(recipient_component.id)
                    recipient_parameter_index = recipient_component.iparams_dict.get(recipient.name)

                    recipient_conn = {
                        'to': parent_instance,
                        'edge': (i, recipient_parameter_index)
                    }
                    recipient_connections.append(recipient_conn)

        # Handle connections from sources to this component's input parameters
        for i, iparam in enumerate(self.iparams):  # Iterate over input parameters
            if iparam.sources:  # Ensure there are sources to consider
                for s in iparam.sources:
                    source = GHParam(s)
                    # Search the canvas for the corresponding objects. Remember to convert the InstanceGUID into a str
                    source_component = canvas.find_object_by_guid(str(source.parent.obj.Attributes.InstanceGuid))
                    source_instance = canvas.find_object_by_guid(source_component.id)
                    source_parameter_index = source_component.oparams_dict.get(
                        source.name)  # Assuming oparams_dict includes source name

                    source_conn = {
                        'from': source_instance,
                        'edge': (source_parameter_index, i)
                    }
                    source_connections.append(source_conn)

        return source_connections, recipient_connections

    def __str__(self):
        return f"{self.name} ({self.category})"

    def __repr__(self):
        return f"<GHComponent {self.__str__()}>"

class Canvas:
    def __init__(self, doc,environment, n=None):
        GHComponentTable.initialise(environment.dirs['vanilla'])
        self.doc = doc
        self.components = []  # Initialize as empty
        self.guid_to_component = {}  # Initialize as empty
        self.initialize_components(n)
        self.graph_id_to_component = self.process_mappings()

    def initialize_components(self, n=None):
        ic("called initialise components")
        if n:
            self.components = [GHComponent(o) for o in list(self.doc.Objects)[:n]]
            self.guid_to_component = {c.id: c for c in self.components}
        else:
            self.components = [GHComponent(o) for o in list(self.doc.Objects)]
            self.guid_to_component = {c.id: c for c in self.components}

    def find_object_by_guid(self, guid: str) -> GHComponent:
        return self.guid_to_component.get(guid)

    def process_mappings(self):
        idx_set = set([component.global_idx for component in self.components])
        idx_lookup = {K : [] for K in idx_set}
        graph_id_to_component = {}
        # ensure there are still incremeting the size of the list
        for component in self.components:
            component_table_idx = GHComponentTable.component_to_idx(component)
            existing_list = idx_lookup[component_table_idx]
            x = (component.global_idx, len(existing_list))
            existing_list.append(x)
            component.graph_id = x
            graph_id_to_component[component.graph_id] = component
        return graph_id_to_component

    def __str__(self):
        return str(self.components)

    def __repr__(self):
        return f"<Canvas {self.__str__()}>"

class GraphConnection:

    def __init__(self, v1n, v1i, v2n, v2i, edge):
        self.v1n = v1n
        self.v1i = v1i
        self.v2n = v2n
        self.v2i = v2i
        self.e1 = edge[0]
        self.e2 = edge[1]
        self.tensor = torch.tensor([self.v1n, self.v1i, self.e1, self.v2n, self.v2i, self.e2], dtype=torch.int16)

    @property
    def edge(self):
        return (self.v1i, self.v1i), (self.v2n, self.v2i)

    @property
    def edgeproperties(self):
        return (self.e1, self.e2)
    def __str__(self):
        return f"GC({self.v1n}-{self.v1i}, {self.v2n}-{self.v2i})"

    def __repr__(self):
        return f"GraphConn ({self.v1n}-{self.v1i}, {self.v2n}-{self.v2i})"

class GraphNode:
    def __init__(self, graph_id: Tuple[int,int], canvas:Canvas):
        self.graph_id: Tuple[int, int] = graph_id
        self.component: GHComponent = canvas.graph_id_to_component.get(graph_id)
        self.canvas: Canvas = canvas

    def edges(self, bidirectional = False):
        """:Returns the edge tuple in the form (int, int), (int,int) where
        the tuple describes the graph node id. If bidirectional is true, both the sources and recipient
        edges are returned, otherwise only the recipient edges are returned"""

        sources, recipients = self.component.get_connections(self.canvas)
        connections = self.get_graph_connections(sources, recipients)
        if bidirectional:
            return connections['sources'] + connections['recipients']
        else:
            return connections['recipients']

    def get_graph_connections(self, sources, recipients):
        graph_connections = {
            "sources": [],
            "recipients": []
        }

        # Process recipient connections (outgoing)
        for connection in recipients:
            v2 = connection.get("to")
            v1n, v1i = self.graph_id
            v2n, v2i = v2.graph_id
            edge = connection.get("edge")
            graph_connections["recipients"].append(GraphConnection(v1n, v1i, v2n, v2i, edge))

        # Process source connections (incoming)
        for connection in sources:
            v1 = connection.get("from")
            v1n, v1i = v1.graph_id
            v2n, v2i = self.graph_id  # For incoming, self.id is the destination
            edge = [int(x) for x in connection.get("edge")]
            graph_connections["sources"].append(GraphConnection(v1n, v1i, v2n, v2i, edge))

        return graph_connections

class GHGraph:
    def __init__(self, canvas):
        self.canvas = canvas
        self.components = self.canvas.components

    @property
    def nodes(self) -> List[GraphNode]:
        return [GraphNode(component.graph_id, self.canvas) for component in self.components]

    def nxGraph(self, bidirectional=False) -> nx.Graph:
        gx = nx.Graph()

        # Step 1: Add all nodes to the graph
        for node in self.nodes:
            node_id = node.graph_id  # Ensure this is a simple, hashable type
            gx.add_node(node_id)  # Explicitly add nodes

        # Step 2: Add edges to the graph
        for node in self.nodes:
            for edge in node.edges():
                if edge:
                    # Ensure the tuples are hashable and correspond to actual node identifiers
                    gx.add_edge((edge.v1n, edge.v1i), (edge.v2n, edge.v2i))

        return gx

    def show_graph(self):
        plt.figure(figsize=(12, 8))
        gx = self.nxGraph()

        # Generate category color map and assign colors
        category_color_map = self.generate_category_color_map()
        node_colors = [category_color_map[node.component.category] for node in self.nodes]

        # Custom labels for nodes
        custom_labels = {node.graph_id: node.component.name for node in self.nodes}

        # Generate positions for all nodes
        pos = nx.spring_layout(gx)

        # Draw the graph with node colors and custom labels
        nx.draw(gx, pos, with_labels=True, labels=custom_labels, node_color=node_colors, node_size=700, cmap=plt.cm.get_cmap('hsv', len(category_color_map)), font_size=5)
        plt.show()

    @staticmethod
    def generate_category_color_map():
        categories = sorted(GHComponentTable.view_all_categories())
        # Create a color map with a fixed set of colors or based on hashing category names
        cmap = plt.cm.get_cmap('tab20b', len(categories))
        category_color_map = {}
        for i, category in enumerate(sorted(categories)):  # Sort categories for consistency
            # Use hashing to ensure consistent color assignment
            hash_val = hash(category) % len(categories)
            color = cmap(hash_val)
            category_color_map[category] = color
        return category_color_map

    def save_graph(self, location):
        gx = self.nxGraph()
        nx.write_graphml(gx, location)

class GHProcessor:
    def __init__(self, filepath_filename, environment):
        self.doc = self.get_ghdoc(filepath_filename)
        self.canvas = Canvas(self.doc, environment)
        self.graph =  GHGraph(self.canvas)
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
        self.graph = GHGraph(self.canvas)
        return self.graph

    def save_graph(self):
        if self.graph is not None:
            self.graph= nx.write_graphml(self.graph, "my_graph.graphml")

def load_create_environment(environment_name):
    # Load or create the environment

    env = EnvironmentManager.create_environment(environment_name)
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

if __name__ == "__main__":
    name = "initial"
    env = load_create_environment(name)
    logging.basicConfig()
    gh_file = next(env.get_gh_file())
    gh_file = str(gh_file)
    ghp = GHProcessor(gh_file, env)
    display(ghp.graph.show_graph())
