import unittest
from gh_graph_extraction import *
from enviroment import *
from gh_logging import filelog

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

        # initialise the environment
        environment = Environment("InitialTest")
        environment.initialise(clone_env=True)
        # TODO: There is a potential issue here because if a person loads an environment but does not copy the
        # TODO components or write the components, there will be a difference between the environment and the components table
        # TODO this needs to  be addressed
        # initialise the grasshopper environment
        comptable = GHComponentTable
        comptable.initialise(environment.dirs['00-VanillaComponents'])
        comptable.to_csv(location=environment.dirs["01-AllComponents"])
        gh_file = environment.get_test_files()
        print(gh_file)
        #initialise the loggers

    #    yeild the next grasshopper file, run the gh processor





    # def test_ben(self):



if __name__ == '__main__':
    unittest.main()
