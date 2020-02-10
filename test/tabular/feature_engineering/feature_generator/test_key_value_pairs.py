import collections
import unittest
import pandas as pd

from src.tabular.feature_engineering.feature_generator.key_value_pairs import key_value_pairs
from sklearn.datasets import load_boston


class key_value_pairs_tester(unittest.TestCase):
    def test_something(self):
        # create config tuple
        configger = collections.namedtuple('configger', ['groupBy_keys', 'groupBy_value', 'method'])

        # config = configger(["RAD", "CHAS"], "B", 'mean')
        # data = load_boston()
        # df = pd.DataFrame(data=data.data, columns=data.feature_names)
        # result = key_value_pairs(df, [config])

        df = pd.DataFrame.from_dict({"A": [1, 2, 1], "B": [1, 1, 1], "C": [3, 4, 5]})
        config = configger(["A", "B"], "C", 'mean')
        result = key_value_pairs(df, [config])
        d = pd.DataFrame.from_dict({"A": [1, 1, 2], "B": [1, 1, 1], "C": [3, 5, 4], "A_B C mean": [4, 4, 4]})
        check_result = d.merge(result, on=["A", "B", "C"])
        self.assertEqual(True, check_result[check_result["A_B C mean_x"] != check_result["A_B C mean_y"]].empty)


if __name__ == '__main__':
    unittest.main()
