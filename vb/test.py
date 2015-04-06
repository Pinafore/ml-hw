import unittest

from numpy import array

from lda import VariationalBayes
from scipy.special import psi as digam
from math import exp

class TestVB(unittest.TestCase):
    def setUp(self):
        self.init_beta = array([[.26, .185, .185, .185, .185],
                                [.185, .185, .26, .185, .185],
                                [.185, .185, .185, .26, .185]])

    def test_phi(self):
        vb = VariationalBayes()

        gamma = array([2.0, 2.0, 2.0])
        beta = self.init_beta
        phi = vb.new_phi(gamma, beta, 0, 1)

        prop = 0.27711205238850234
        normalizer = sum(x * prop for x in beta[:, 0])
        self.assertAlmostEqual(phi[0], beta[0][0] * prop / normalizer)
        self.assertAlmostEqual(phi[1], beta[1][0] * prop / normalizer)
        self.assertAlmostEqual(phi[2], beta[2][0] * prop / normalizer)

    def test_m(self):
        vb = VariationalBayes()
        vb.init([], "stuck", 3)
        # vb.m_step(self.init_beta)

        # self.assertAlmostEqual(self.init_beta[2][3], vb._beta[2][3])

        topic_count = array([[5., 4., 3., 2., 1.],
                             [0., 2., 2., 4., 1.],
                             [1., 1., 1., 1., 1.]])

        vb.m_step(topic_count)
        self.assertAlmostEqual(vb._beta[2][3], .2)
        self.assertAlmostEqual(vb._beta[0][0], .33333333)
        self.assertAlmostEqual(vb._beta[1][4], .11111111)
        self.assertAlmostEqual(vb._beta[0][3], .13333333)

if __name__ == '__main__':
    unittest.main()
