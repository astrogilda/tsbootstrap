# Adopted and modified from https://github.com/fracdiff/fracdiff/tree/main/fracdiff/sklearn

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
from typing import Optional, Tuple
from functools import partial
import numpy as np

from scipy.special import binom  # type: ignore


def fdiff_coef(d: float, window: int) -> np.ndarray:
    """Returns sequence of coefficients in fracdiff operator.

    Parameters
    ----------
    d : float
        Order of differentiation.
    window : int
        Number of terms.

    Returns
    -------
    coef : numpy.array, shape (window,)
        Coefficients in fracdiff operator.

    Examples
    --------
    >>> fdiff_coef(0.5, 4)
    array([ 1.    , -0.5   , -0.125 , -0.0625])
    >>> fdiff_coef(1.0, 4)
    array([ 1., -1.,  0., -0.])
    >>> fdiff_coef(1.5, 4)
    array([ 1.    , -1.5   ,  0.375 ,  0.0625])
    """
    return (-1) ** np.arange(window) * binom(d, np.arange(window))


def fdiff(
    a: np.ndarray,
    n: float = 1.0,
    axis: int = 0,
    prepend: Optional[np.ndarray] = None,
    append: Optional[np.ndarray] = None,
    window: int = 10,
    mode: str = "same",
) -> np.ndarray:
    """Calculate the `n`-th differentiation along the given axis.

    Extention of ``numpy.diff`` to fractional differentiation.

    Parameters
    ----------
    a : array_like
        The input array.
    n : float, default=1.0
        The order of differentiation.
        If ``n`` is an integer, returns the same output with ``numpy.diff``.
    axis : int, default=-1
        The axis along which differentiation is performed, default is the first axis.
    prepend : array_like, optional
        Values to prepend to ``a`` along axis prior to performing the differentiation.
        Scalar values are expanded to arrays with length 1 in the direction of axis and
        the shape of the input array in along all other axes.
        Otherwise the dimension and shape must match ``a`` except along axis.
    append : array_like, optional
        Values to append.
    window : int, default=10
        Number of observations to compute each element in the output.
    mode : {"same", "valid"}, default="same"
        "same" (default) :
            At the beginning of the time series,
            return elements where at least one coefficient of fracdiff is used.
            Output size along ``axis`` is :math:`L_{\\mathrm{in}}`
            where :math:`L_{\\mathrm{in}}` is the length of ``a`` along ``axis``
            (plus the lengths of ``append`` and ``prepend``).
            Boundary effects may be seen at the at the beginning of a time-series.
        "valid" :
            Return elements where all coefficients of fracdiff are used.
            Output size along ``axis`` is
            :math:`L_{\\mathrm{in}} - \\mathrm{window} + 1` where
            where :math:`L_{\\mathrm{in}}` is the length of ``a`` along ``axis``
            (plus the lengths of ``append`` and ``prepend``).
            Boundary effects are not seen.

    Returns
    -------
    fdiff : numpy.ndarray
        The fractional differentiation.
        The shape of the output is the same as ``a`` except along ``axis``.

    Examples
    --------
    This returns the same result with ``numpy.diff`` for integer `n`.

    >>> from fracdiff import fdiff
    >>> a = np.array([1, 2, 4, 7, 0])
    >>> (np.diff(a) == fdiff(a)).all()
    True
    >>> (np.diff(a, 2) == fdiff(a, 2)).all()
    True

    This returns fractional differentiation for noninteger `n`.

    >>> fdiff(a, 0.5, window=3)
    array([ 1.   ,  1.5  ,  2.875,  4.75 , -4.   ])

    Mode "valid" returns elements for which all coefficients are convoluted.

    >>> fdiff(a, 0.5, window=3, mode="valid")
    array([ 2.875,  4.75 , -4.   ])
    >>> fdiff(a, 0.5, window=3, mode="valid", prepend=[1, 1])
    array([ 0.375,  1.375,  2.875,  4.75 , -4.   ])

    Differentiation along desired axis.

    >>> a = np.array([[  1,  3,  6, 10, 15],
    ...               [  0,  5,  6,  8, 11]])
    >>> fdiff(a, 0.5, window=3)
    array([[1.   , 2.5  , 4.375, 6.625, 9.25 ],
           [0.   , 5.   , 3.5  , 4.375, 6.25 ]])
    >>> fdiff(a, 0.5, window=3, axis=0)
    array([[ 1. ,  3. ,  6. , 10. , 15. ],
           [-0.5,  3.5,  3. ,  3. ,  3.5]])
    """
    if mode == "full":
        mode = "same"
        raise DeprecationWarning("mode 'full' was renamed to 'same'.")

    if isinstance(n, int) or n.is_integer():
        prepend = np._NoValue if prepend is None else prepend  # type: ignore
        append = np._NoValue if append is None else append  # type: ignore
        return np.diff(a, n=int(n), axis=axis, prepend=prepend, append=append)

    if a.ndim == 0:
        raise ValueError(
            "diff requires input that is at least one dimensional")

    a = np.asanyarray(a)
    # Mypy complains:
    # fracdiff/fdiff.py:135: error: Module has no attribute "normalize_axis_index"
    axis = np.core.multiarray.normalize_axis_index(
        axis, a.ndim)  # type: ignore
    dtype = a.dtype if np.issubdtype(a.dtype, np.floating) else np.float64

    combined = []
    if prepend is not None:
        prepend = np.asanyarray(prepend)
        if prepend.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            prepend = np.broadcast_to(prepend, tuple(shape))
        combined.append(prepend)

    combined.append(a)

    if append is not None:
        append = np.asanyarray(append)
        if append.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            append = np.broadcast_to(append, tuple(shape))
        combined.append(append)

    if len(combined) > 1:
        a = np.concatenate(combined, axis)

    if mode == "valid":
        D = partial(np.convolve, fdiff_coef(
            n, window).astype(dtype), mode="valid")
        a = np.apply_along_axis(D, axis, a)
    elif mode == "same":
        # Convolve with the mode 'full' and cut last
        D = partial(np.convolve, fdiff_coef(
            n, window).astype(dtype), mode="full")
        s = tuple(
            slice(a.shape[axis]) if i == axis else slice(None) for i in range(a.ndim)
        )
        a = np.apply_along_axis(D, axis, a)
        a = a[s]
    else:
        raise ValueError("Invalid mode: {}".format(mode))

    return a


class Fracdiff(BaseEstimator, TransformerMixin):
    def __init__(self, d: float = 1.0, window: int = 10, mode: str = "same", window_policy: str = "fixed"):
        """
        Initialize Fracdiff transformer.

        Parameters
        ----------
        d : float
            Order of differentiation.
        window : int
            Number of terms.
        mode : str
            Convolution mode. One of {"full", "valid", "same"}.
        window_policy : str
            Window size policy. One of {"fixed", "increasing"}.
        """
        self.d = self.validate_d(d)
        self.window = self.validate_window(window)
        self.mode = self.validate_mode(mode)
        self.window_policy = self.validate_window_policy(window_policy)

    def __repr__(self) -> str:
        """
        String representation of the Fracdiff transformer.

        Returns
        -------
        str
            String representation of the Fracdiff transformer.
        """
        name = self.__class__.__name__
        params = ", ".join(f"{attr}={getattr(self, attr)}" for attr in [
                           "d", "window", "mode", "window_policy"])
        return f"{name}({params})"

    def validate_d(self, d: float) -> float:
        if not isinstance(d, float) or d < 0:
            raise ValueError("d must be a non-negative float.")
        return d

    def validate_window(self, window: int) -> int:
        if not isinstance(window, int) or window <= 0:
            raise ValueError("window must be a positive integer.")
        return window

    def validate_mode(self, mode: str) -> str:
        valid_modes = ["full", "valid", "same"]
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}.")
        return mode

    def validate_window_policy(self, window_policy: str) -> str:
        valid_policies = ["fixed", "increasing"]
        if window_policy not in valid_policies:
            raise ValueError(f"window_policy must be one of {valid_policies}.")
        return window_policy

    def fit(self, X: np.array, y: Optional[np.array] = None) -> 'Fracdiff':
        """
        Compute the coefficients of fractional differentiation operator.

        Parameters
        ----------
        X : ndarray
            Input array.
        y : ndarray, optional
            Target array.

        Returns
        -------
        Fracdiff
            The fitted transformer.
        """
        X = check_array(X)
        self.X_copy_ = X.copy()
        self.diff_coef_ = fdiff_coef(self.d, self.window)
        return self

    def transform(self, X: np.array, y: Optional[np.array] = None) -> Tuple[np.array, np.array]:
        """
        Apply fractional differentiation to the array.

        Parameters
        ----------
        X : ndarray
            Input array.
        y : ndarray, optional
            Target array.

        Returns
        -------
        Tuple[ndarray, ndarray]
            Fractionally differentiated array and the initial values.
        """
        check_is_fitted(self, "diff_coef_")
        X = check_array(X, estimator=self)
        self.initial_values_ = self.extract_initial_values(X)
        X_diff = fdiff(X, n=self.d, axis=0, window=self.window, mode=self.mode)
        return X_diff

    def extract_initial_values(self, X: np.array) -> np.array:
        """
        Extract the initial values from the array.

        Parameters
        ----------
        X : ndarray
            Input array.

        Returns
        -------
        ndarray
            Initial values.
        """
        if X.shape[0] < self.window:
            raise ValueError(
                f"Number of time steps, {X.shape[0]}, must be greater than or equal to window size, {self.window}.")
        return X[:self.window, :]

    def inverse_transform(self, X: np.array) -> np.array:
        """
        Apply inverse fractional differentiation to the array.

        Parameters
        ----------
        X : ndarray
            Input array.
        initial_values : ndarray
            The initial values of the array before fractional differentiation.

        Returns
        -------
        ndarray
            The array after inverse fractional differentiation.
        """
        check_is_fitted(self, "diff_coef_")
        X = check_array(X)
        initial_values = check_array(self.initial_values_)

        if self.mode == "valid" and X.shape[0] != self.X_copy_.shape[0] - self.window + 1:
            raise ValueError(
                f"For 'valid' mode, X.shape[0] must be equal to X_copy_.shape[0] - window + 1. Got X.shape[0] = {X.shape[0]}, X_copy_.shape[0] = {self.X_copy_.shape[0]}, and window = {self.window}.")
        elif self.mode == "same" and X.shape[0] != self.X_copy_.shape[0]:
            raise ValueError(
                f"For 'same' mode, X.shape[0] must be equal to X_copy_.shape[0] + window - 1. Got X.shape[0] = {X.shape[0]}, X_copy_.shape[0] = {self.X_copy_.shape[0]}, and window = {self.window}.")

        return self.undiff(X, initial_values)

    def undiff(self, X: np.array, initial_values: np.array) -> np.array:
        """
        Apply inverse fractional differentiation to the array.

        Parameters
        ----------
        X : ndarray
            Input array.
        initial_values : ndarray
            The initial values of the array before fractional differentiation.

        Returns
        -------
        ndarray
            The array after inverse fractional differentiation.
        """
        if self.mode == "valid":
            # Calculate the number of initial values lost due to 'valid' mode
            lost_values = self.window - 1
            # Replace the lost values with the initial ones
            initial_values = initial_values[:, :lost_values]

        padded_X = np.concatenate((initial_values, X), axis=0)
        return np.apply_along_axis(lambda x: np.convolve(x, self.diff_coef_, mode="valid"), 0, padded_X)
