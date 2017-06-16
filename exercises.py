import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from statsmodels.stats import outliers_influence

def chapter_3():
    """
    Notes for Linear Regression
    
    - coefficients -> give average change in Y with a one-unit increase in X
    
    - confidence interval -> B1_hat +- 2 * SE(B1_hat)
        - 95% chance the interval contains true value of B
        - SE(B1_hat) -> var(e) / SSE
    
    - t-statistic
        - t = (B1_hat - 0)/ SE(B1_hat)
        
    -  test for synergy (additive assumption)
        - effect of each predictor on response is independent of other predictors
        - include interaction term -> x1 * x2 
            - if interaction term has small p value, then not additive (synergy exists)
            - if results in substantial increase in r2, then not additive (synergy exists)

    - relationship exists
        - p value < 0.0005 or < 0.0001
        - F statistic greater than 1

    - strength of relationship
        - RSE -> estimates standard deviation of response from regression line
        - R squared -> % variability in response explained by predictors
        - percent error -> 100 * residual_standard_error / ys.mean()  
        
    - accuracy of prediction
        - prediction interval (individual response)
        - confidence interval (average response)
        
    - non-linearity
        - residual plots (fitted values vs. studentized/standardized residuals)
        - if residual plots are not random, transform with log(x), sqrt(x), or x2
        
    - correlation of error terms
        - will underestimate p value and narrow confidence/prediction intervals
        
    - heteroscedasticity (funnel shape of residual plot)
        - non-constant variances in the errors
        - if exists, transform the response with log(y) or sqrt(y)
    
    - co-linearity of features
        - (VIF) variance inflation factor -> 1 / (1 - r2)
        - correlation matrix
        - reduces t-statistic and increases standard error
        
    - outliers
        - leverage -> high impact on RSE and/or regression line
        - look at studentized residuals (observations > 3 are outliers)
        - influence (leverage) plots
    """
    #3.8 -> Simple Linear Regression on Auto data set
    dat = pd.read_csv("Auto.csv")
    dat = dat.replace("?", np.nan).dropna()

    # add constant to x values to ensure mean of residuals = 0
    xs = sm.add_constant(dat["horsepower"].astype(float))
    ys = dat["mpg"].astype(float)

    model = sm.OLS(ys, xs).fit()

    intercept, slope = model.params
    r2 = model.rsquared
    
    # variance inflation factor -> test for co-linearity
    # min(VIF) = 1.0, if VIF > 5 or 10, features are most likely correlated
    vif = 1 / (1 - r2)
    f_stat = model.fvalue
    p_value = model.pvalues[1]
    
    # create new line with the coefficients
    fit = [slope*x + intercept for x in xs["horsepower"]]
    print("Simple OLS: %s" % model.summary())

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
                                       
    # plot fitted values vs residuals to check for non-linearity
    ax2.scatter(model.fittedvalues, residuals, color="r")
    ax2.axhline(0, color="k")
    ax2.set_xlabel("fitted values")
    ax2.set_ylabel("residuals")
    
    # show leverage to identity observations that may have
    # more effect on the regression than other observations
    sm.graphics.influence_plot(model, ax=ax3)
    
    # show fitted values vs studentized residuals 
    outlier_influence = outliers_influence.OLSInfluence(model).summary_frame()
    ax4.scatter(model.fittedvalues, outlier_influence["student_resid"])
    ax4.axhline(0, color="k")
    ax4.set_xlabel("fitted values")
    ax4.set_ylabel("studentized residuals")
    
    for _ax in [ax, ax2, ax3, ax4]:
        _ax.legend(loc="best")
    plt.show()
    
    #3.9 -> Multiple Linear Regression on Auto data set 
    xs = dat[["cylinders", "displacement", "horsepower", 
              "weight", "acceleration", "year", "origin"]].astype(float)
              
    # plot correlation matrix to check co-linearity
    # co-linearity reduces the t-statistic (power) of the test
    # and also increases standard error
    print("Correlations: %s" % xs.corr())    
    grid = sns.PairGrid(xs)
    grid = grid.map(plt.scatter)
    plt.show()

    results = pd.ols(y=ys, x=xs)
    model = sm.OLS(ys, xs).fit()
    print("Multiple OLS: %s" % results)
    
    # compute variance inflation factor (VIF) to check for co-linearity
    vif = list(map(lambda x: 1 / (1 - x), model.params))
    print("VIFs: %s" % vif)

    """
    Looking at the p-values associated with each predictor’s t-statistic, 
    we see that displacement, weight, year, and origin 
    have a statistically significant relationship, 
    while cylinders, horsepower, and acceleration do not.
    """
    print("Coefficients: %s" % results.beta)
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
    
def chapter_4():
    '''
    Notes for Logistic Regression:
    
    - coefficients -> give the change in log odds with a one-unit increase in X
    '''

    
    
    
    
if __name__ == "__main__":
    chapter_3()
