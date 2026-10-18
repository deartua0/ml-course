import unittest
from unittest.mock import patch

import numpy as np

from bagging import SimplifiedBaggingRegressor


class MeanRegressor:
    def fit(self, data, target):
        self.data = data.copy()
        self.target = target.copy()
        self.mean = np.mean(target)
        return self

    def predict(self, data):
        if len(data) == 0:
            raise ValueError('Empty prediction input.')
        return np.full(len(data), self.mean)


class BaggingTests(unittest.TestCase):
    def setUp(self):
        self.data = np.array([[1], [2], [3], [4]])
        self.target = np.array([1, 2, 7, 8])
        self.bags = [np.array(bag) for bag in (
            [1, 3, 1, 0], [2, 2, 0, 3], [0, 1, 3, 3]
        )]

    def fit_example(self, data=None, oob=True):
        regressor = SimplifiedBaggingRegressor(3, oob=oob)
        with patch('numpy.random.choice', side_effect=self.bags):
            regressor.fit(MeanRegressor, self.data if data is None else data, self.target)
        return regressor

    def test_independent_models_and_paired_rows(self):
        regressor = self.fit_example()
        self.assertEqual(len({id(model) for model in regressor.models_list}), 3)
        for model, indices in zip(regressor.models_list, self.bags):
            np.testing.assert_array_equal(model.data, self.data[indices])
            np.testing.assert_array_equal(model.target, self.target[indices])
        np.testing.assert_allclose(regressor.predict(self.data[:2]), [55 / 12, 55 / 12])

    def test_oob_example_averages_predictions_before_errors(self):
        regressor = self.fit_example()
        for model, prediction in zip(regressor.models_list, [6, 3, 8]):
            model.mean = prediction
        self.assertEqual(regressor.OOB_score(), 0.5)
        self.assertEqual([list(p) for p in regressor.list_of_predictions_lists],
                         [[], [3], [6, 8], []])
        self.assertEqual(list(regressor.oob_predictions), [None, 3, 7, None])

    def test_identical_features_still_have_distinct_indices(self):
        regressor = self.fit_example(data=np.ones((4, 1)))
        self.assertEqual(regressor.OOB_score(), 11.53125)
        self.assertEqual(list(regressor.oob_predictions), [None, 5.75, 4, None])

    def test_no_oob_objects(self):
        regressor = SimplifiedBaggingRegressor(1, oob=True)
        regressor.fit(MeanRegressor, self.data[:1], self.target[:1])
        with self.assertRaisesRegex(ValueError, 'No objects'):
            regressor.OOB_score()
        self.assertEqual(list(regressor.oob_predictions), [None])

    def test_oob_disabled(self):
        regressor = self.fit_example(oob=False)
        with self.assertRaisesRegex(ValueError, 'oob=True'):
            regressor.OOB_score()


if __name__ == '__main__':
    unittest.main()
