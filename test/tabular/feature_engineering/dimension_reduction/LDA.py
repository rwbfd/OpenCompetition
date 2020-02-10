import unittest
import pandas as pd
from src.tabular.feature_engineering.dimension_reduction.LDA import LDA
from collections import namedtuple
from sklearn.datasets import load_iris

class LDA_test(unittest.TestCase):
    def test_something(self):
        d = load_iris()
        df = pd.DataFrame(d.data, columns=d.feature_names)
        df.loc[:, "target"] = d.target

        configger = namedtuple('config', ['reduce_col','target_col', 'n_components'])
        config = configger(None, "target",2)
        result = LDA(df,config)
        print(result)

        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
