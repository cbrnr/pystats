from numpy.testing import assert_almost_equal
import pandas as pd
from pyanova import anova


def test_plantgrowth():
    data = pd.read_csv("http://bit.ly/plantgrowthdata",
                       names=["id", "weight", "group"], header=0)
    r = anova(data, "weight", "id", between=["group"], within=None)

    group = [3.76634, 10.49209, 2, 27, 4.84609, 0.01591]
    assert_almost_equal(r.loc["group", "SSn":"p"], group, 5)


def test_toothgrowth():
    data = pd.read_csv("http://bit.ly/toothgrowthdata",
                       names=["id", "len", "supp", "dose"], header=0)
    r = anova(data, "len", "id", between=["supp", "dose"], within=None)

    supp = [205.35, 712.106, 1, 54, 15.57198, 2.31183e-04]
    dose = [2426.43434, 712.106, 2, 54, 91.99996, 4.0463e-18]
    s_d = [108.319, 712.106, 2, 54, 4.10699, 2.18603e-02]
    assert_almost_equal(r.loc["supp", "SSn":"p"], supp, 5)
    assert_almost_equal(r.loc["dose", "SSn":"p"], dose, 5)
    assert_almost_equal(r.loc["supp:dose", "SSn":"p"], s_d, 5)


def test_toy():
    data = pd.read_csv("pyanova/tests/toy.csv")
    r = anova(data, "v", "id", between=["a", "b", "c"], within=None)

    a = [2.5, 127.6, 1, 32, 0.62696, 0.43430]
    b = [0.1, 127.6, 1, 32, 0.02508, 0.87517]
    c = [4.9, 127.6, 1, 32, 1.22884, 0.27590]
    ab = [0.1, 127.6, 1, 32, 0.02508, 0.87517]
    ac = [62.5, 127.6, 1, 32, 15.67398, 0.00039]
    bc = [2.5, 127.6, 1, 32, 0.62696, 0.43430]
    abc = [0.9, 127.6, 1, 32, 0.22571, 0.63795]
    assert_almost_equal(r.loc["a", "SSn":"p"], a, 5)
    assert_almost_equal(r.loc["b", "SSn":"p"], b, 5)
    assert_almost_equal(r.loc["c", "SSn":"p"], c, 5)
    assert_almost_equal(r.loc["a:b", "SSn":"p"], ab, 5)
    assert_almost_equal(r.loc["a:c", "SSn":"p"], ac, 5)
    assert_almost_equal(r.loc["b:c", "SSn":"p"], bc, 5)
    assert_almost_equal(r.loc["a:b:c", "SSn":"p"], abc, 5)

