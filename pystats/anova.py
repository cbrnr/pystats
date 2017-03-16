import numpy as np
from numpy.linalg import lstsq
import pandas as pd
from itertools import chain, combinations
from functools import reduce
from collections import OrderedDict
from scipy.stats import f


def anova(data, dv, idv, between=None, within=None):
    """Perform analysis of variance (ANOVA).

    Parameters
    ----------
    data : pd.DataFrame
        Input data frame (long format).
    dv : str
        Dependent variable (column name).
    idv : str
        Case identifier (column name).
    between : str or list of str
        Factor(s) that was/were manipulated between cases.
    within : str or list of str
        Factor(s) that was/were manipulated within cases.

    Returns
    -------
    r : pd.DataFrame
        Table containing detailed results.
    """
    y = data[dv]

    if between is None and within is None:
        raise ValueError("No between or within factors specified.")

    if not isinstance(between, list):
        between = [between]
    if not isinstance(within, list):
        within = [within]

    # between += [idv]
    dummies = [_dummies(data[b]) for b in between]

    intercept = [np.ones((len(y), 1))]
    contrasts = [_contrasts(d) for d in dummies]
    interactions = _interactions(contrasts)

    coefs = intercept + contrasts + interactions
    names = ["1"] + [":".join(n) for n in list(_powerset(between))]
    terms = OrderedDict(zip(names, coefs))

    model = []
    ss = []  # sums of squares
    df = []  # degrees of freedom
    for name, term in terms.items():
        model.append(term)
        x = np.column_stack([*model])
        b = lstsq(x, y)[0]  # regression coefficients
        yhat = x.dot(b)  # predicted values
        ss.append(np.sum((yhat - y) ** 2))  # residual sum of squares
        dfs = [terms[e].shape[1] for e in name.split(":")]  # df of each effect
        df.append(reduce(lambda a, b: a * b, dfs))  # df of interaction

    r = pd.DataFrame({"SSn": np.abs(-np.diff(np.array(ss)))}, index=names[1:])
    r["SSd"] = ss[-1]
    r["DFn"] = df[1:]
    r["DFd"] = (len(y) - 1) - sum(df[1:])  # total df minus full model df
    r["F"] = (r["SSn"] / r["DFn"]) / (r["SSd"] / r["DFd"])  # F statistic
    r["p"] = 1 - f.cdf(r["F"], r["DFn"], r["DFd"])  # p-value
    r["sig"] = r["p"].apply(_sig)  # significance
    return r


def _powerset(x):
    """Return the powerset of a given list of items.

    Parameters
    ----------
    x : list
        Input list.

    Returns
    -------
    p : itertools.chain
        Powerset (set of all possible subsets).

    Example
    -------
    >>> list(_powerset([1, 2, 3]))
    [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    """
    return chain.from_iterable(combinations(x, r) for r in range(1, len(x)+1))


def _dummies(x):
    """Create dummy coding variable(s) from categorical input.

    Parameters
    ----------
    x : np.array
        Categorical input variable.

    Returns
    -------
    d : np.array
        Dummy-coded output variable(s).

    Example
    -------
    >>> _dummies(np.array(["a", "a", "b", "a", "c"]))
    array([[ 1.,  0.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  1.]])
    """
    return (x[:, None] == np.unique(x)).astype(float)


def _contrasts(x):
    """Create contrasts.

    Parameters
    ----------
    x : np.array
        Input variables (usually dummy-coded).

    Returns
    -------
    c : np.array
        Contrasts (using the first column as the baseline).

    Example
    -------
    >>> d = _dummies(np.array(["a", "a", "b", "a", "c"]))
    >>> _contrasts(d)
    array([[-1., -1.],
           [-1., -1.],
           [ 1.,  0.],
           [-1., -1.],
           [ 0.,  1.]])
    """
    return x[:, 1:] - x[:, :1]


def _interactions(x):
    """Calculate interaction terms.

    Parameters
    ----------
    x : list
        List of contrasts.

    Returns
    -------
    i : list
        List of interaction terms.

    Example
    -------
    >>> d = _dummies(np.array(["a", "a", "b", "a", "c"]))
    >>> c = _contrasts(d)
    >>> _interactions([c, c])
    [array([[ 1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.],
           [ 1.,  0.,  0.,  0.],
           [ 1.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  1.]])]

    """
    i = list(_powerset(range(len(x))))[len(x):]
    return [reduce(lambda a, b: _mult(a, b), [x[y] for y in idx]) for idx in i]


def _mult(x, y):
    """Multiply each column from x with each column from y.
    """
    return (y[:, None] * x[..., None]).reshape(x.shape[0], -1)


def _sig(p):
    """Mark significant results.

    Parameters
    ----------
    p : float
        Number (usually a p-value).

    Returns
    -------
    s : str
        Depending on the magnitude of the input, the result is either "***",
        "**", "*" or "".
    """
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


if __name__ == "__main__":
    data = pd.read_csv("tests/bushtucker.csv")
    r = anova(data, "value", "participant", between="variable")
    print(r)
