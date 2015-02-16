import unittest
from svm import weight_vector, find_support, find_slack, kINSP, kSEP
from numpy import array, zeros


class TestSVM(unittest.TestCase):
    def setUp(self):
        self.sep_x = kSEP[:, 0:2]
        self.sep_y = kSEP[:, 2]
        self.insep_x = kINSP[:, 0:2]
        self.insep_y = kINSP[:, 2]

    def test_wide_slack(self):
        w = array([-.25, .25])
        b = -.25
        self.assertEqual(find_slack(self.insep_x, self.insep_y, w, b),
                         set([6, 4]))

    def test_narrow_slack(self):
        w = array([0, 2])
        b = -5

        self.assertEqual(find_slack(self.insep_x, self.insep_y, w, b),
                         set([3, 5]))

    def test_support(self):
        w = array([0.2, 0.8])
        b = -0.2

        self.assertEqual(find_support(self.sep_x, self.sep_y, w, b),
                         set([0, 4, 2]))

    def test_weight(self):
        alpha = zeros(len(self.sep_x))
        alpha[4] = 0.34
        alpha[0] = 0.12
        alpha[2] = 0.22

        w = weight_vector(self.sep_x, self.sep_y, alpha)
        self.assertAlmostEqual(w[0], 0.2)
        self.assertAlmostEqual(w[1], 0.8)

if __name__ == '__main__':
    unittest.main()
