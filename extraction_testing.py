import unittest
from gh_graph_extraction import *

class MyTestCase(unittest.TestCase):

    def test_canvas(self):
        canvas = Canvas(d)
        self.assertEqual(len(canvas.components), 139)  # add assertion here
        print(canvas.guid_to_component.items())



if __name__ == '__main__':
    unittest.main()
