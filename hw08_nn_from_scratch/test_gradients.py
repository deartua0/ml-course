"""Numerical gradient and integration checks requiring only NumPy."""
import unittest

import numpy as np

from modules import (
    BatchNormalization, ClassNLLCriterion, ClassNLLCriterionUnstable,
    Dropout, ELU, LeakyReLU, Linear, LogSoftMax, ReLU, Sequential,
    SoftMax, SoftPlus, adam_optimizer,
)


def numerical_gradient(function, values, epsilon=1e-6):
    result = np.zeros_like(values)
    for index in np.ndindex(values.shape):
        original = values[index]
        try:
            values[index] = original + epsilon
            plus = function()
            values[index] = original - epsilon
            minus = function()
        finally:
            values[index] = original
        result[index] = (plus - minus) / (2 * epsilon)
    return result


class TestGradients(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.rng = np.random.default_rng(42)

    def test_layer_input_gradients(self):
        layers = [SoftMax(), LogSoftMax(), LeakyReLU(), ELU(), SoftPlus(),
                  BatchNormalization()]
        evaluation_bn = BatchNormalization()
        evaluation_bn.forward(self.rng.normal(size=(6, 4)))
        evaluation_bn.evaluate()
        layers.append(evaluation_bn)
        for layer in layers:
            with self.subTest(layer=repr(layer), training=layer.training):
                values = self.rng.normal(size=(6, 4))
                incoming = self.rng.normal(size=values.shape)
                layer.forward(values)
                actual = layer.backward(values, incoming).copy()
                expected = numerical_gradient(
                    lambda: np.sum(layer.forward(values) * incoming), values)
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    def test_network_parameter_and_input_gradients(self):
        network = Sequential()
        first, last = Linear(2, 3), Linear(3, 2)
        for layer in [first, ReLU(), last]:
            network.add(layer)
        values = self.rng.normal(size=(4, 2))
        incoming = self.rng.normal(size=(4, 2))
        network.forward(values)
        self.assertGreater(np.min(np.abs(first.output)), 1e-4)
        network.zeroGradParameters()
        actual_input = network.backward(values, incoming).copy()
        arrays = [(values, actual_input)]
        for layer in [first, last]:
            arrays.extend((parameter, gradient.copy()) for parameter, gradient
                          in zip(layer.getParameters(), layer.getGradParameters()))
        for values_to_change, actual in arrays:
            expected = numerical_gradient(
                lambda: np.sum(network.forward(values) * incoming), values_to_change)
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    def test_loss_gradients(self):
        target = np.eye(3)[[0, 1, 2, 0]]
        cases = [(ClassNLLCriterionUnstable(), self.rng.uniform(.1, .9, (4, 3))),
                 (ClassNLLCriterion(), self.rng.normal(size=(4, 3)))]
        for criterion, values in cases:
            with self.subTest(criterion=repr(criterion)):
                actual = criterion.backward(values, target).copy()
                expected = numerical_gradient(lambda: criterion.forward(values, target), values)
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)

    def test_numerical_stability_and_softmax_cross_entropy(self):
        values = np.array([[-10000., 0., 10000.], [10000., 9999., -10000.]])
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            for layer in [SoftMax(), LogSoftMax(), ELU(), SoftPlus()]:
                output = layer.forward(values)
                gradient = layer.backward(values, np.ones_like(values))
                self.assertTrue(np.isfinite(output).all())
                self.assertTrue(np.isfinite(gradient).all())
        layer = SoftMax()
        values = self.rng.normal(size=(5, 3))
        probabilities = layer.forward(values).copy()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        np.testing.assert_allclose(layer.forward(values + 1000), probabilities, atol=1e-13)
        target = np.eye(3)[[0, 1, 2, 0, 1]]
        layer.forward(values)
        loss = ClassNLLCriterionUnstable()
        actual = layer.backward(values, loss.backward(probabilities, target))
        np.testing.assert_allclose(actual, (probabilities - target) / len(values), atol=1e-14)

    def test_dropout_modes_and_mask(self):
        values = np.ones((1000, 10))
        for probability in [0., .5, 1.]:
            layer = Dropout(probability)
            output = layer.forward(values)
            np.testing.assert_array_equal(layer.backward(values, values), output)
            if probability == .5:
                self.assertLess(abs(output.mean() - 1), .05)
            layer.evaluate()
            np.testing.assert_array_equal(layer.forward(values), values)
            np.testing.assert_array_equal(layer.backward(values, values), values)

    def test_end_to_end_learning(self):
        network = Sequential()
        for layer in [Linear(2, 12), ReLU(), Linear(12, 2), LogSoftMax()]:
            network.add(layer)
        values = self.rng.normal(size=(128, 2))
        target = np.eye(2)[(np.sum(values ** 2, axis=1) < 1).astype(int)]
        criterion = ClassNLLCriterion()
        initial = criterion.forward(network.forward(values), target)
        config = dict(learning_rate=.02, beta1=.9, beta2=.999, epsilon=1e-8)
        state = {}
        for _ in range(300):
            output = network.forward(values)
            network.zeroGradParameters()
            network.backward(values, criterion.backward(output, target))
            adam_optimizer(network.getParameters(), network.getGradParameters(), config, state)
        final = criterion.forward(network.forward(values), target)
        accuracy = np.mean(network.output.argmax(axis=1) == target.argmax(axis=1))
        self.assertLess(final, initial * .5)
        self.assertGreater(accuracy, .9)


if __name__ == '__main__':
    unittest.main(verbosity=2)
