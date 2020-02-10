import unittest
from sklearn.datasets import load_boston
import pandas as pd
from src.tabular.feature_engineering.feature_generator.symbolic_learning import GPConfig,symbolicLearning


class MyTestCase(unittest.TestCase):
    def test_something(self):
        d = load_boston()
        df = pd.DataFrame(d.data, columns=d.feature_names)
        df.loc[:, "target"] = d.target
        gpconfiger = GPConfig(["CRIM", "ZN"], "target")
        res = symbolicLearning(df, gpconfiger)
        print(res)

        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()



