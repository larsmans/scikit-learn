# Authors: Fabian Pedregosa
#          Alexandre Gramfort
#          Lars Buitinck
# License: 3-clause BSD

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from .base import LinearClassifierMixin, SparseCoefMixin
from ..base import BaseEstimator
from ..feature_selection.from_model import _LearntSelectorMixin
from ..preprocessing import LabelBinarizer, LabelEncoder
from ..svm.base import BaseLibLinear
from ..utils import (atleast2d_or_csr, check_arrays, column_or_1d,
                     compute_class_weight)
from ..utils.extmath import logsumexp, safe_sparse_dot


class _LogRegMixin(LinearClassifierMixin):
    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        return self._predict_proba_lr(X)

    def predict_log_proba(self, X):
        """Log of probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """
        return np.log(self.predict_proba(X))


class LogisticRegression(BaseLibLinear, _LogRegMixin, _LearntSelectorMixin,
                         SparseCoefMixin):
    """Logistic Regression (aka logit, MaxEnt) classifier.

    In the multiclass case, the training algorithm uses a one-vs.-all (OvA)
    scheme, rather than the "true" multinomial LR.

    This class implements L1 and L2 regularized logistic regression using the
    `liblinear` library. It can handle both dense and sparse input. Use
    C-ordered arrays or CSR matrices containing 64-bit floats for optimal
    performance; any other input format will be converted (and copied).

    Parameters
    ----------
    penalty : string, 'l1' or 'l2'
        Used to specify the norm used in the penalization.

    dual : boolean
        Dual or primal formulation. Dual formulation is only
        implemented for l2 penalty. Prefer dual=False when
        n_samples > n_features.

    C : float, optional (default=1.0)
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.

    fit_intercept : bool, default: True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added the decision function.

    intercept_scaling : float, default: 1
        when self.fit_intercept is True, instance vector x becomes
        [x, self.intercept_scaling],
        i.e. a "synthetic" feature with constant value equals to
        intercept_scaling is appended to the instance vector.
        The intercept becomes intercept_scaling * synthetic feature weight
        Note! the synthetic feature weight is subject to l1/l2 regularization
        as all other features.
        To lessen the effect of regularization on synthetic feature weight
        (and therefore on the intercept) intercept_scaling has to be increased

    class_weight : {dict, 'auto'}, optional
        Over-/undersamples the samples of each class according to the given
        weights. If not given, all classes are supposed to have weight one.
        The 'auto' mode selects weights inversely proportional to class
        frequencies in the training set.

    tol: float, optional
        Tolerance for stopping criteria.

    Attributes
    ----------
    `coef_` : array, shape = [n_classes, n_features]
        Coefficient of the features in the decision function.

        `coef_` is readonly property derived from `raw_coef_` that \
        follows the internal memory layout of liblinear.

    `intercept_` : array, shape = [n_classes]
        Intercept (a.k.a. bias) added to the decision function.
        If `fit_intercept` is set to False, the intercept is set to zero.

    random_state: int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use when
        shuffling the data.

    See also
    --------
    LinearSVC

    Notes
    -----
    The underlying C implementation uses a random number generator to
    select features when fitting the model. It is thus not uncommon,
    to have slightly different results for the same input data. If
    that happens, try with a smaller tol parameter.

    References:

    LIBLINEAR -- A Library for Large Linear Classification
        http://www.csie.ntu.edu.tw/~cjlin/liblinear/

    Hsiang-Fu Yu, Fang-Lan Huang, Chih-Jen Lin (2011). Dual coordinate descent
        methods for logistic regression and maximum entropy models.
        Machine Learning 85(1-2):41-75.
        http://www.csie.ntu.edu.tw/~cjlin/papers/maxent_dual.pdf


    See also
    --------
    sklearn.linear_model.MultinomialLR
    sklearn.linear_model.SGDClassifier
    """

    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None):

        super(LogisticRegression, self).__init__(
            penalty=penalty, dual=dual, loss='lr', tol=tol, C=C,
            fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
            class_weight=class_weight, random_state=random_state)


class MultinomialLR(BaseEstimator, _LogRegMixin):
    """Multinomial logistic regression.

    This class implements logistic regression for multiclass problems. While
    LogisticRegression is able to do multiclass classification out of the box,
    the probabilities it estimates are not well-calibrated for such problems
    as it fits one binary model per class. By contrast, multinomial LR
    estimators solve a single multiclass optimization problem and minimize
    the cross-entropy (aka. log loss) over the whole probability
    distribution P(y=k|X).

    Parameters
    ----------
    alpha : float, optional
        Strength of L2 penalty (aka. regularization, weight decay).
        Note that the (optional) intercept term is not regularized.

    class_weight : {dict, 'auto'}, optional
        Over-/undersamples the samples of each class according to the given
        weights. If not given, all classes are supposed to have weight one.
        The 'auto' mode selects weights inversely proportional to class
        frequencies in the training set.

    fit_intercept : bool, default: True
        Whether an intercept (bias) term should be learned and added to the
        decision function.

    Attributes
    ----------
    `coef_` : array, shape = [n_classes, n_features]
        Coefficient of the features in the decision function.

    `intercept_` : array, shape = [n_classes]
        Intercept (a.k.a. bias) added to the decision function.
        If `fit_intercept` is set to False, the intercept is set to zero.

    References
    ----------
    C. M. Bishop (2006). Pattern Recognition and Machine Learning. Springer,
        pp. 205-210.

    See also
    --------
    sklearn.linear_model.LogisticRegression
    sklearn.linear_model.SGDClassifier

    """
    def __init__(self, alpha=1e-4, fit_intercept=True, class_weight=None):
        self.alpha = alpha
        self.class_weight = class_weight
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """Fit logistic regression model to training data X, y.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target vector for the samples in X.
        """

        X = atleast2d_or_csr(X, dtype=float)
        y = column_or_1d(y, warn=True)
        X, y = check_arrays(X, y)

        self._lb = LabelBinarizer()
        Y = self._lb.fit_transform(y)
        if Y.shape[1] == 1:
            Y = np.hstack([1 - Y, Y])

        # compute_class_weight cannot handle negative integers, so
        # transform to (0, n_classes - 1)
        y_enc = LabelEncoder().fit_transform(y)
        self.class_weight_ = compute_class_weight(
            self.class_weight, self.classes_, y_enc)
        if self.class_weight:
            print self.class_weight_
            Y = Y * self.class_weight_ / np.sum(self.class_weight_)

        # Fortran-ordered so we can slice off the intercept in loss_grad and
        # get contiguous arrays.
        w = np.zeros((Y.shape[1], X.shape[1] + bool(self.fit_intercept)),
                     order='F')

        C = 1. / self.alpha
        if C < 0:
            raise ValueError("Penalty term must be positive; got (alpha=%r)"
                             % self.alpha)
        w, loss, info = fmin_l_bfgs_b(_loss_grad, w.ravel(),
                                      args=[X, Y, C, self.fit_intercept])
        w = w.reshape(Y.shape[1], -1)
        if self.fit_intercept:
            intercept = w[:, -1]
            w = w[:, :-1]
        else:
            intercept = np.zeros(Y.shape[1])

        if len(self.classes_) == 2:
            w = w[1].reshape(1, -1)
            intercept = intercept[1:]
        self.coef_ = w
        self.intercept_ = intercept

        return self

    @property
    def classes_(self):
        return self._lb.classes_


def _loss_grad(w, X, Y, C, fit_intercept):
    # Cross-entropy loss and its gradient for multinomial logistic regression
    # (Bishop 2006, p. 209) with L2 penalty (weight decay).

    # Used for regularisation later.
    l2_ = np.dot(w, w)

    w = w.reshape(Y.shape[1], -1)
    if fit_intercept:
        intercept = w[:, -1]
        w = w[:, :-1]
    else:
        intercept = 0

    p = safe_sparse_dot(X, w.T)
    p += intercept
    p -= logsumexp(p, axis=1).reshape(-1, 1)
    loss = (-C * (Y * p).sum()) + .5 * (l2_ - np.dot(intercept, intercept))

    p = np.exp(p, p)
    diff = p - Y
    grad = safe_sparse_dot(diff.T, X)
    grad *= C
    grad += w
    if fit_intercept:
        grad = np.hstack([grad, diff.sum(axis=0).reshape(-1, 1)])

    return loss, grad.ravel()
