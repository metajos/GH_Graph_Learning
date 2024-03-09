import logging
import unittest
from environment_manager import *


class MyTestCase(unittest.TestCase):
    def setUp(self):
        # This method is called before each test method runs.
        self.name = "240308-initial"
        self.env = load_create_environment(
            self.name)  # Assuming load_create_environment is a function you've defined
        self.file = "210309_Canopy Modelling.gh"
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
        canvas = Canvas(self, doc, self.env)
        graph = GHGraph(canvas)
        edgs = []
        for node in graph.nodes:
            print(node.component.name)
            edges = node.edges()
            edgs.append(edges)
        print(edgs)

    def test_nxGraph(self):
        doc = GHProcessor.get_ghdoc(str(self.gh_file))
        canvas = Canvas(self, doc, self.env)
        graph = GHGraph(canvas)
        nxgraph = graph.nxGraph()
        self.assertTrue(nxgraph is not None)
    def test_save_graph(self):
        doc = GHProcessor.get_ghdoc(str(self.gh_file))
        canvas = Canvas(self, doc, self.env)
        graph = GHGraph(canvas)
        nxgraph = graph.nxGraph()
        location = Path(r"C:\Users\jossi\Dropbox\Office_Work\Jos\GH_Graph_Learning\test") / self.gh_file.stem
        graph.save_graph(str(location))
        display(graph.show_graph(str(str(location) + ".png")))


    # def test_all_files(self):
    #     test_multiple(self.env)


if __name__ == '__main__':
    unittest.main()
