
from dan import DeepAveragingNetwork, roll_params
from numpy import array

import unittest

class TestDan(unittest.TestCase):
    def setUp(self):
        self.dan = DeepAveragingNetwork(2, 2, 4, 2, 2, .5)
        params = self.dan._params
        # params.append(array([[ 8,  2,  -3,  4 ],
        #                      [-5,  8,  -3, -9]]))
        # params.append(array([5, -7]))        
        # params.append(array([[ 1, -1],
        #                      [ 0,  2]]))
        # params.append(array([1, -2]))
        # params.append(array([[ -2, 1],
        #                      [  1, 3]]))
        # params.append(array([ .5, -.5]))
        # params.append(array([[  1, 4],
        #                      [  2, -8]]))
        
        self.dan._params = params
        
    def test_activation(self):
        activiation = self.dan.activations([0, 1])

        self.assertAlmostEqual(activation[0], .2)

if __name__ == '__main__':
    unittest.main()
