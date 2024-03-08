import logging
import unittest

from environment_manager import *

class MyTestCase(unittest.TestCase):

    def test_canvas(self):
        canvas = Canvas(d)
        # self.assertEqual(len(canvas.components), 139)  # add assertion here

        graph =  GHGraph(canvas)
        display(graph.show_graph())

    def test_GHProcessor(self):
        pass
        # filepath = os.getcwd()
        # filename = "test.gh"
        # ghp = GHProcessor(filepath, filename)
        # display(ghp.graph.show_graph())
        # # graph =  GHGraph(canvas)
        # # display(graph.show_graph())

    def test_graph(self):
        filepath = os.getcwd()
        filename = "test_grafting.gh"
        ghp = GHProcessor(filepath, filename)
        display(ghp.graph.show_graph())

    def test_environments(self):

        name = "initial"
        env = load_create_environment(name)
        logging.basicConfig()
        gh_file =  next(env.get_gh_file())
        gh_file = str(gh_file)
        ghp = GHProcessor(gh_file, env)
        display(ghp.graph.show_graph())









    # def test_ben(self):



if __name__ == '__main__':


    unittest.main()
