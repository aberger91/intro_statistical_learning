import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns


def chapter_3():
    #3.8 -> Simple Linear Regression on Auto data set
    dat = pd.read_csv("Auto.csv")
    dat = dat.replace("?", np.nan).dropna()

    xs = sm.add_constant(dat["horsepower"].astype(float))
    ys = dat["mpg"].astype(float)

    model = sm.OLS(ys, xs)
    model = model.fit()

    intercept, slope = model.params
    r2 = model.rsquared
    f_stat = model.fvalue
    p_value = model.pvalues[1]
    fit = [slope*x + intercept for x in xs["horsepower"]]
    print(model.summary())

    prediction = model.predict()
    residuals = ys.astype(float) - prediction
    standardized_residuals = (residuals - residuals.mean()) / \
                             (residuals.max() - residuals.min())
    #residual_standard_error = results.rmse
    #percent_error = 100 * residual_standard_error / ys.mean()        

    """
    Plot
    """
    f = plt.figure()
    ax = f.add_subplot(221)
    ax2 = f.add_subplot(223)
    ax3 = f.add_subplot(222)
    ax4 = f.add_subplot(224)
    ax.scatter(xs["horsepower"], ys, label="r2=%f; f=%f; p=%f" % (
                              r2, f_stat, p_value))
    ax.plot(xs["horsepower"], fit, color="r", label="f(x) = %f * x + %f" % (
                                       slope, intercept))
    # non-linear if this plot is not a random scatter plot
    ax2.scatter(xs["horsepower"], residuals, color="r")
    ax2.axhline(0, color="k")
    sm.graphics.influence_plot(model, ax=ax3)
    ax4.scatter(prediction, standardized_residuals, label="prediction vs. standardized residuals")
    ax4.axhline(0, color="k")
    for _ax in [ax, ax2, ax3, ax4]:
        _ax.legend(loc="best")
    plt.show()
    
    #3.9 -> Multiple Linear Regression on Auto data set 
    xs = dat[["cylinders", "displacement", "horsepower", 
              "weight", "acceleration", "year", "origin"]].astype(float)
    grid = sns.PairGrid(xs)
    grid = grid.map(plt.scatter)
    plt.show()        
    
    print(dat.corr())
    results = pd.OLS(y=ys, x=xs)
    model = sm.OLS(ys, xs)
    print(results)

    """
    Looking at the p-values associated with each predictor’s t-statistic, 
    we see that displacement, weight, year, and origin 
    have a statistically significant relationship, 
    while cylinders, horsepower, and acceleration do not.
    """
    print(results.beta)    
    """
    The regression coefficient for year, 0.7508, 
    suggests that for every one year, mpg increases by the coefficient. 
    In other words, cars become more fuel efficient every year by almost 1 mpg / year.
    """  
    residuals = results.resid
    standardized_residuals = (residuals - residuals.mean()) / \
                             (residuals.max() - residuals.min())
    residual_standard_error = results.rmse
    percent_error = 100 * residual_standard_error / ys.mean()  
    
    """
    Plot
    """
    f = plt.figure()
    ax = f.add_subplot(221)
    ax2 = f.add_subplot(223)
    ax3 = f.add_subplot(222)
    ax4 = f.add_subplot(224)
    ax.scatter(results.y_fitted, residuals)
    ax.axhline(0, color="k")
    ax.set_xlabel("y fitted values")
    ax.set_ylabel("residuals")
    ax2.scatter(results.y_fitted, standardized_residuals, label='percent error=%f' % percent_error)
    ax2.axhline(0, color="k")
    ax2.set_xlabel("y fitted values")
    ax2.set_ylabel("standardized residuals")
    sm.graphics.influence_plot(model, ax=ax3)
 
    for _ax in [ax, ax2, ax3, ax4]:
        _ax.legend(loc="best")
    plt.show()

if __name__ == "__main__":
    chapter_3()
