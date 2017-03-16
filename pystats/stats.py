import numpy as np
import scipy.stats
from collections import namedtuple


def sd(x):
    """Compute unbiased standard deviation.

    Parameters
    ----------
    x : array
        Input data.

    Returns
    -------
    sd : array
        Standard deviation.
    """
    return np.std(x, ddof=1)


def ttest(x, y, paired=True):
    """Compute t-test.

    Parameters
    ----------
    x : array
        First input variable.
    y : array
        Second input variable.
    paired : bool, optional
        Perform paired t-tests (default) or unpaired t-tests.

    Returns
    -------
    t : float
        Value of t statistic.
    df : float
        Degrees of freedom.
    p : float
        Significance (p-value).
    d : float
        Mean difference.
    cl : float
        Lower confidence boundary.
    cu : float
        Upper confidence boundary.
    """
    if paired:
        n = len(x)
        df = n - 1
        t, p = scipy.stats.ttest_rel(x, y)
        delta = x - y
        d = np.mean(delta)
        cl = d - scipy.stats.t.ppf(0.975, df) * sd(delta)/np.sqrt(n)
        cu = d + scipy.stats.t.ppf(0.975, df) * sd(delta)/np.sqrt(n)
    else:
        raise NotImplementedError  # TODO unpaired t-test
    Ttest = namedtuple("Ttest", ["t", "df", "p", "d", "cl", "cu"])
    return Ttest(t, df, p, d, cl, cu)


def cor(x, y):
    """Compute Pearson correlation coefficient.

    Parameters
    ----------
    x : array
        First input variable.
    y : array
        Second input variable.

    Returns
    -------
    r : float
        Pearson correlation coefficient.
    t : float
        Value of t statistic.
    df : float
        Degrees of freedom.
    p : float
        Significance (p-value).
    cl : float
        Lower confidence boundary.
    cu : float
        Upper confidence boundary.
    """
    n = len(x)
    df = n - 2
    r, p = scipy.stats.pearsonr(x, y)
    t = r * np.sqrt((len(x) - 2) / (1 - r**2))
    # confidence interval taken from R cor.test()
    z = np.arctanh(r)
    sigma = 1 / np.sqrt(n - 3)
    cl = np.tanh(z - sigma * scipy.stats.norm.ppf(0.975))
    cu = np.tanh(z + sigma * scipy.stats.norm.ppf(0.975))
    Cor = namedtuple("Cor", ["r", "t", "df", "p", "cl", "cu"])
    return Cor(r, t, df, p, cl, cu)
