import unittest
import numpy as np
import pandas as pd
from collections import namedtuple
from src.tabular.feature_engineering.feature_generator.encoding_continus_variable import encode_continuous_variable


class continus_variable_encode_tester(unittest.TestCase):
    def test_something(self):
        configger = namedtuple('config', ['encode_col', 'method'])
        # x = np.arange(-0.99, 1, 0.01)
        x = np.random.normal(0, 1, size=10000)
        from scipy.stats import kstest
        print(kstest(rvs=x, cdf="norm"))
        print(kstest(rvs=x, cdf="uniform"))
        df = pd.DataFrame(x, columns=["t"])
        config = configger("t", "MinMax")
        result = encode_continuous_variable(df, [config])
        print(result)
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
