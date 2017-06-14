import numpy as np
import statsmodels.api as sm
import pandas as pd
dat = pd.read_csv("Auto.csv")
dat = dat.replace("?", np.nan).dropna()
xs = sm.add_constant(dat["horsepower"].astype(float))
ys = dat["mpg"].astype(float)
model = sm.OLS(ys, xs)
fit = model.fit()
print(fit.summary())
r2 = fit.rsquared
print(r2)
intercept, slope = fit.params
p_value = fit.pvalues[1]
f_stat = fit.fvalue
print(intercept, slope, p_value, f_stat)
print(xs)
