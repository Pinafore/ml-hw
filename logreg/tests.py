import unittest

from logreg import LogReg, Example

kTOY_VOCAB = "BIAS_CONSTANT A B C D".split()
kPOS = Example(1, "A:4 B:3 C:1".split(), kTOY_VOCAB, None)
kNEG = Example(0, "B:1 C:3 D:4".split(), kTOY_VOCAB, None)

class TestKnn(unittest.TestCase):
    def setUp(self):
        self.logreg_unreg = LogReg(5, 0.0, lambda x: 1.0)
        self.logreg_reg = LogReg(5, 0.25, lambda x: 1.0)

    def test_reg(self):
        print(self.logreg_reg.beta)
        print(kPOS.x)
        beta = self.logreg_reg.sg_update(kPOS, 0)
        self.assertAlmostEqual(beta[0], .25)
        self.assertAlmostEqual(beta[1], 1.0)
        self.assertAlmostEqual(beta[2], 0.75)
        self.assertAlmostEqual(beta[3], 0.25)
        self.assertAlmostEqual(beta[4], 0.0)

        print(self.logreg_reg.beta)
        print(kPOS.x)
        beta = self.logreg_reg.sg_update(kNEG, 1)
        self.assertAlmostEqual(beta[0], -0.30097640098415529)
        self.assertAlmostEqual(beta[1], 1.0)
        self.assertAlmostEqual(beta[2], -0.05097640098415529)
        self.assertAlmostEqual(beta[3], -1.1529292029524658)
        self.assertAlmostEqual(beta[4], -0.85195280196831058)

    def test_unreg(self):
        print(self.logreg_unreg.beta)
        print(kPOS.x)
        beta = self.logreg_unreg.sg_update(kPOS, 0)
        self.assertAlmostEqual(beta[0], .5)
        self.assertAlmostEqual(beta[1], 2.0)
        self.assertAlmostEqual(beta[2], 1.5)
        self.assertAlmostEqual(beta[3], 0.5)
        self.assertAlmostEqual(beta[4], 0.0)

        print(self.logreg_unreg.beta)
        print(kPOS.x)
        beta = self.logreg_unreg.sg_update(kNEG, 1)
        self.assertAlmostEqual(beta[0], -0.47068776924864364)
        self.assertAlmostEqual(beta[1], 2.0)
        self.assertAlmostEqual(beta[2], 0.5293122307513564)
        self.assertAlmostEqual(beta[3], -2.4120633077459308)
        self.assertAlmostEqual(beta[4], -3.8827510769945746)

if __name__ == '__main__':
    unittest.main()
