import unittest
from gh_graph_extraction import *
from enviroment import *

class MyTestCase(unittest.TestCase):

    def test_canvas(self):
        canvas = Canvas(d)
        # self.assertEqual(len(canvas.components), 139)  # add assertion here

        graph =  GHGraph(canvas)
        display(graph.show_graph())

    def test_GHProcessor(self):
        filepath = os.getcwd()
        filename = "test.gh"
        ghp = GHProcessor(filepath, filename)
        display(ghp.graph.show_graph())
        # graph =  GHGraph(canvas)
        # display(graph.show_graph())

    def test_graph(self):
        filepath = os.getcwd()
        filename = "test_grafting.gh"
        ghp = GHProcessor(filepath, filename)
        display(ghp.graph.show_graph())

    def test_environments(self):
        environment = Environment("Test1")

    # def test_ben(self):



if __name__ == '__main__':
    unittest.main()
