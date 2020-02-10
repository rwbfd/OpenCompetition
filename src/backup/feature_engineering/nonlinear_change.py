from scipy import stats,special
import pandas as pd
import numpy as np


data = pd.read('data.csv')
y = data.target

lam_range = np.linspace(-2,5,100)  # default nums=50
llf = np.zeros(lam_range.shape, dtype=float)

# lambda estimate:
for i,lam in enumerate(lam_range):
    llf[i] = stats.boxcox_llf(lam, y)		# y 必须>0

# find the max lgo-likelihood(llf) index and decide the lambda
lam_best = lam_range[llf.argmax()]

#对预测变量进行cox-box变换
y_boxcox = special.boxcox1p(y, lam_best)

#对预测变量进行逆cox-box变换
y_invboxcox = special.inv_boxcox1p(y_boxcox, lam_best)