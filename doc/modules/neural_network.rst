.. _neural_networks:

===============
Neural networks
===============

.. currentmodule:: sklearn.neural_network

Neural networks are estimators and models inspired by the way
the human brain learns and processes information.


Multilayer perceptrons
----------------------

Multilayer perceptrons (MLPs) are one of the basic neural network architectures
and can be used for predictive modeling,
i.e. classification and regression.
MLPs are purely feedforward networks:
test samples are passed through a series of layers,
each depending only on the information it received from earlier steps.

More specifically, the MLP in scikit-learn computes the
``decision_function``

.. math::
    f(x) = \sigma(x W_1^\top + b_1) W_2^\top + b_2

where :math:`x` is a feature vector
and :math:`\sigma` is a non-linear activation function.
By default, :math:`\sigma` is the hyperbolic tangent, a sigmoid function.
In the case of classification, the ``argmax`` of this function is taken;
probabilities are computed by a softmax.

Training of MLPs amounts to finding optimal weight matrices and bias vectors
:math:`W_1, b_1, W_2, b_2`
by means either stochastic gradient descent (SGD) or limited-memory BFGS.
Gradients are computed using the backpropagation (backprop) algorithm.

.. note::
    scikit-learn's multilayer perceptrons are currently limited
    to a single hidden layer.

:class:`MLPClassifier` implements a multi-layer perceptron for classification,
trained under a log-loss regime (aka cross-entropy error),
the same error function used in multinomial logistic regression.

:class:`MLPRegressor` implements the same type of network for regression,
trained under mean squared error (MSE) loss.
