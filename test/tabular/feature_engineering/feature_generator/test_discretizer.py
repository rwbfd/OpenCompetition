import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from collections import namedtuple
from src.tabular.feature_engineering.feature_generator.discretizer import discretizer

class discretizer_test(unittest.TestCase):
    def test_something(self):
        configger = namedtuple('config', ['encode_col', 'method',"n_bins","target_col"])
        data = load_breast_cancer()

        df = pd.DataFrame(data.data,columns=data.feature_names)
        df.loc[:,"target"] = data.target
        config = configger(["mean radius","mean texture"],"trees",5,"target")
        result = discretizer(df,[config])
        print(result)
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
