import unittest

from rademacher import origin_plane_hypotheses, axis_aligned_hypotheses, \
    rademacher_estimate, kSIMPLE_DATA as rad_data, PlaneHypothesis, \
    constant_hypotheses

from vc_sin import train_sin_classifier


def assign_exists(data, classifiers, pattern):
    """
    Given a dataset and set of classifiers, make sure that the classification
    pattern specified exists somewhere in the classifier set.
    
    This is testing whether a particular dichotomyis realized in your set of classifiers.  One of the elements in the hyps argument should produce the classification result in pattern.
    """

    val = False
    assert len(data) == len(pattern), "Length mismatch between %s and %s" % \
        (str(data), str(pattern))
    for hh in classifiers:
        present = all(hh.classify(data[x]) == pattern[x] for
                      x in xrange(len(data)))
        # Uncomment for additional debugging code
        # if present:
        #    print("%s matches %s" % (str(hh), str(pattern)))
        val = val or present
    if not val:
        print("%s not found in:" % str(pattern))
        for hh in classifiers:
            print("\t%s %s" % (str(hh), [hh.classify(x) for x in data]))
    return val


class TestLearnability(unittest.TestCase):
    def setUp(self):
        self._2d = {}
        self._2d[1] = [(3, 3)]
        self._2d[2] = [(3, 3), (3, 4)]
        self._2d[3] = [(3, 3), (3, 4), (4, 3)]
        self._2d[4] = rad_data

        self._hypotheses = lambda x: [PlaneHypothesis(0, 0, 5),
                                      PlaneHypothesis(0, 0, -5),
                                      PlaneHypothesis(0, 1, 0),
                                      PlaneHypothesis(0, -1, 0),
                                      PlaneHypothesis(1, 0, 0),
                                      PlaneHypothesis(-1, 0, 0)]
        self._full_shatter = [(1, 1), (-1, -1)]
        self._half_shatter = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    def test_rec_single_point(self):
        hyps = list(axis_aligned_hypotheses(self._2d[1]))
        self.assertTrue(assign_exists(self._2d[1], hyps, [True]))
        self.assertTrue(assign_exists(self._2d[1], hyps, [False]))

    def test_rec_two_points(self):
        hyps = list(axis_aligned_hypotheses(self._2d[2]))
        for pp in [[False, False], [False, True], [True, False], [True, True]]:
            self.assertTrue(assign_exists(self._2d[2], hyps, pp))

    def test_rec_three_points(self):
        hyps = list(axis_aligned_hypotheses(self._2d[3]))
        for pp in [[False, False, False],
                   [False, True, False], [True, False, False], [False, False, True],
                   [True, True, False], [True, False, True],
                   [True, True, True]]:
            self.assertTrue(assign_exists(self._2d[3], hyps, pp))

    def test_rec_four_points(self):
        hyps = list(axis_aligned_hypotheses(self._2d[4]))
        self.assertEqual(14, len(hyps))

    def test_plane_four_points(self):
        hyps = list(origin_plane_hypotheses(self._2d[4]))

        for pp in [[True, True, True, True],
                   [False, False, True, True],
                   [False, False, True, False],
                   [True, True, False, True],
                   [True, True, False, False],
                   [False, False, False, False]]:
            self.assertTrue(assign_exists(self._2d[4], hyps, pp))

        self.assertEqual(6, len(hyps))

    def test_correlation(self):
        labels = [+1, +1, -1, +1]
        hyp = PlaneHypothesis(0, 1, 0)

        self.assertEqual(hyp.correlation(self._half_shatter, labels), -.5)

    def test_rad_estimate(self):
        self.assertAlmostEqual(1.0, rademacher_estimate(self._full_shatter,
                                                        self._hypotheses,
                                                        num_samples=1000,
                                                        random_seed=3),
                               places=1)

        self.assertAlmostEqual(0.0, rademacher_estimate([(0, 0)],
                                                        constant_hypotheses,
                                                        num_samples=1000,
                                                        random_seed=3),
                               places=1)

        self.assertAlmostEqual(0.625, rademacher_estimate(self._half_shatter,
                                                          self._hypotheses,
                                                          num_samples=1000,
                                                          random_seed=3),
                               places=1)


    def test_vc_one_point_pos(self):
        data_pos = [(1, False)]

        classifier_pos = train_sin_classifier(data_pos)

        for xx, yy in data_pos:
            self.assertEqual(True if yy == +1 else False,
                             classifier_pos.classify(xx))

    def test_vc_one_point_neg(self):
        data_neg = [(1, True)]

        classifier_neg = train_sin_classifier(data_neg)

        for xx, yy in data_neg:
            self.assertEqual(True if yy == +1 else False,
                             classifier_neg.classify(xx))

    def test_vc_three_points(self):
        data = [(1, False), (2, True), (3, False)]
        classifier = train_sin_classifier(data)

        for xx, yy in data:
            self.assertEqual(True if yy == +1 else False,
                             classifier.classify(xx))

    def test_vc_four_points(self):
        data = [(1, False), (2, True), (3, False), (5, False)]
        classifier = train_sin_classifier(data)

        for xx, yy in data:
            self.assertEqual(True if yy == +1 else False,
                             classifier.classify(xx))

if __name__ == '__main__':
    unittest.main()
