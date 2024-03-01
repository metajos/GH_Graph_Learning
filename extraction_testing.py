import unittest
from gh_graph_extraction import *

class MyTestCase(unittest.TestCase):

    def test_canvas(self):
        canvas = Canvas(d)
        # self.assertEqual(len(canvas.components), 139)  # add assertion here

        graph =  GHGraph(canvas)
        display(graph.show_graph())




if __name__ == '__main__':
    unittest.main()
