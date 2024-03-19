import logging
import unittest
from environment_manager import *


class MyTestCase(unittest.TestCase):
    def setUp(self):
        # This method is called before each test method runs.
        self.name = "240308-initial"
        self.env = load_create_environment(
            self.name)  # Assuming load_create_environment is a function you've defined
        self.file = "test.gh"
        self.gh_file = self.env.dirs[
                           "files"] / self.file  # Assuming env.dirs["files"] gives a path that supports '/' operator

    def test_GHComponent(self):
        # Use self.gh_file to access the gh_file variable
        doc = GHProcessor.get_ghdoc(
            str(self.gh_file))  # Assuming GHProcessor.get_ghdoc is a function you've defined
        for obj in doc.Objects:
            try:
                components = GHComponent(obj)  # Assuming GHComponent is a function/class you've defined
            except Exception as e:
                print(f"{obj.Name}: error")
                raise Exception

    def test_Canvas(self):
        # Use self.gh_file and self.env to access the variables
        doc = GHProcessor.get_ghdoc(str(self.gh_file))
        name = self.gh_file.stem
        canvas = Canvas(name, doc, self.env)  # Assuming Canvas is a function/class you've defined
        self.assertTrue(canvas.components is not None)
        self.assertTrue(len(canvas.components) > 1)
        print(f"#Components: {len(canvas.components)}")


    def test_Graph(self):
        doc = GHProcessor.get_ghdoc(str(self.gh_file))
        canvas = Canvas(self, doc, self.env)
        graph = GHGraph(canvas)
        graphnodes = graph.nodes
        print(f"#GraphNodes: {len(graphnodes)}")

    def test_Edges(self):
        doc = GHProcessor.get_ghdoc(str(self.gh_file))
        canvas = Canvas(self.name, doc, self.env)
        graph = GHGraph(canvas)
        edgs = []
        for node in graph.nodes:
            print(node.component.name)
            edges = node.edges()
            edgs.append(edges)
        print(edgs)

    def test_nxGraph(self):
        doc = GHProcessor.get_ghdoc(str(self.gh_file))
        canvas = Canvas(self.name, doc, self.env)
        graph = GHGraph(canvas)
        nxgraph = graph.nxGraph()
        self.assertTrue(nxgraph is not None)
    def test_save_graph(self):
        doc = GHProcessor.get_ghdoc(str(self.gh_file))
        canvas = Canvas(self.name, doc, self.env)
        graph = GHGraph(canvas)
        nxgraph = graph.nxGraph()
        location = self.gh_file
        graph.save_graph(str(location))
        graph.graph_img(str(str(location) + ".png"))


    # def test_all_files(self):
    #     test_multiple(self.env)

    def test_illegals(self):
        doc = GHProcessor.get_ghdoc(str(self.gh_file))
        illegals = ["Panel"]
        canvas = Canvas(self, doc, self)


    def test_graph_processing(self):
        name = "240308-initial_test"
        env = load_create_environment(name)
        GHComponentTable.initialise()
        filepath = Path(r"C:\Users\jossi\Dropbox\Office_Work\Jos\GH_Graph_Learning\TTD\bigboy\01533-Bridge Alignment-Option 1.gh")
        doc = GHProcessor.get_ghdoc(str(filepath))
        canvas = Canvas("canvas", doc, env)
        gh_graph = GHGraph(canvas)
        # for node in gh_graph.nodes:
        #     print(node.pos)
        gh_graph.save_graph(env.dirs["graphml"] / filepath.stem)
        gh_graph.graph_img(env.dirs["graphml"] / filepath.stem)

    def test_graph_feature_vector(self):
        name = "240308-initial_test"
        env = load_create_environment(name)
        GHComponentTable.initialise()
        filepath = Path(r"C:\Users\jossi\Dropbox\Office_Work\Jos\GH_Graph_Learning\TTD\bigboy\01533-Bridge Alignment-Option 1.gh")
        doc = GHProcessor.get_ghdoc(str(filepath))
        canvas = Canvas("canvas", doc, env)
        gh_graph = GHGraph(canvas)
        for node in gh_graph.nodes:
            print(node.feature_vector)


    def test_exporting_folder(self):
        filelogger = create_named_logger("file_logger", "file_logger.log")
        name = "240318-initial parsing"
        illegals_dict = {
            "Bifocals": "aced9701-8be9-4860-bc37-7e22622afff4",
            "Group": "c552a431-af5b-46a9-a8a4-0fcbc27ef596",
            "Sketch": "2844fec5-142d-4381-bd5d-4cbcef6d6fed",
            "Cluster": "f31d8d7a-7536-4ac8-9c96-fde6ecda4d0a",
            "Scribble": "7f5c6c55-f846-4a08-9c9a-cfdc285cc6fe"
        }
        env = load_create_environment(name)
        GHComponentTable.initialise()
        folder = Path(env.dirs['processed'])
        error_bin = Path(env.dirs['logs'])
        files = list(folder.glob("*.gh"))
        for i, file in enumerate(files):
            print(f"Preprocessing {file.name}...")
            ghpp = GHDocumentPreprocessor(str(file), error_bin)
            filelogger.info(f"processing: {file}")

            doc =ghpp.process_folder_or_file(illegals_dict)
            try:
                canvas = Canvas("canvas", doc, env)
                gh_graph = GHGraph(canvas)
                gh_graph.save_graph(env.dirs["graphml"] / file.stem)
                gh_graph.graph_img(env.dirs["graphml"] / file.stem, show=False)
                filelogger.info(f"processed : {i}/{len(files)} : {file}")
            except Exception as e:
                filelogger.warning(f"ERROR : {i}/{len(files)} : {file}")
                ghpp.move_file_to_error_bin(file, e)
            print("error" + str(file))

            print("done" + str(file))

    def test_preprocessing(self):
        name = "240318-initial parsing"
        env = load_create_environment(name)
        GHComponentTable.initialise()
        folder = Path(env.dirs['files'])



if __name__ == '__main__':
    unittest.main()
