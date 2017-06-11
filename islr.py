import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import datasets
import quandl as qdl
import seaborn as sns
from quandl_products import *

qproducts = QUANDL_PRODUCTS
api_key_path_ = "C:/Users/Andrew/api/"

try:
    qdl.ApiConfig.api_key = open(api_key_path_ + "quandl.txt", "r").read()
except Exception as e:
    print("could not register quandl key\n%s" % e)

def get_quandl_data(product, 
                    start="2015-01-01", 
                    end="2018-01-01"):
    """
    provides a short-hand lookup for products
    you do not need to know the database, only the product code
    """
    if product not in qproducts:
        _product = "WIKI/%s" % product
    else:
        _product = qproducts[product]
    df = qdl.get(_product, 
                 start_date=start, 
                 end_date=end)
    return df

def n_neighbors(n=15):
    from sklearn import neighbors
    from matplotlib.colors import ListedColormap

    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features. We could
                          # avoid this ugly slicing by using a two-dim dataset
    y = iris.target

    h = .02  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    for weights in ['uniform', 'distance']:
        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(n, weights=weights)
        clf.fit(X, y)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification (k = %i, weights = '%s')"
                  % (n, weights))
    plt.show()

def binom_pmf(n=100, p=0.5):
    # There are n+1 possible number of "successes": 0 to n.
    x = range(n+1)
    # k successes in n trials with p probability of success
    y = stats.binom.pmf(x, n, p)
    plt.bar(x, y)
    plt.title("Binomial distribution PMF n=%f & p=%f" % (n, p))
    plt.xlabel("Variate")
    plt.ylabel("Probability")
    plt.show()
    
def binom_cdf(n=100, p=0.5):
    # There are n+1 possible number of "successes": 0 to n.
    x = range(n+1)
    # k successes in n trials with p probability of success
    y = stats.binom.cdf(x, n, p)
    plt.plot(x, y)
    plt.title("Cumulative Binomial distribution n=%f & p=%f" % (n, p))
    plt.xlabel("Variate")
    plt.ylabel("Probability")
    plt.show()
    
def norm_pmf(n=100, mean=0.0, std=1.0):
    x = np.linspace(-n/10, n/10, n)
    y = stats.norm.pdf(x, loc=mean, scale=std)
    plt.bar(x, y)
    plt.xlabel("Variate")
    plt.ylabel("Gaussian Probability Density Function")
    plt.title("CDF for Gaussian of mean=%f & std=%f" % (mean, std))
    plt.show()
    
def norm_cdf(n=100, mean=0.0, std=1.0):
    x = np.linspace(-n/10, n/10, n)
    y = stats.norm.cdf(x, loc=mean, scale=std)
    plt.plot(x, y)
    plt.xlabel("Variate")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF for Gaussian of mean=%f & std=%f" % (mean, std))
    plt.show()
    
def simulate_poisson():
    # Mean is 1.69
    mu = 1.69
    sigma = sp.sqrt(mu)
    mu_plus_sigma = mu + sigma

    # Draw random samples from the Poisson distribution, to simulate
    # the observed events per 2 second interval.
    counts = stats.poisson.rvs(mu, size=100)

    # Bins for the histogram: only the last bin is closed on both
    # sides. We need one more bin than the maximum value of count, so
    # that the maximum value goes in its own bin instead of getting
    # added to the previous bin.
    # [0,1), [1, 2), ..., [max(counts), max(counts)+1]
    bins = range(0, max(counts)+2)

    # Plot histogram.
    plt.hist(counts, bins=bins, align="left", histtype="step", color="black")

    # Create Poisson distribution for given mu.
    x = range(0,10)
    prob = stats.poisson.pmf(x, mu)*100 

    # Plot the PMF.
    plt.plot(x, prob, "o", color="black")

    # Draw a smooth curve through the PMF.
    l = sp.linspace(0,11,100)
    s = interpolate.spline(x, prob, l)
    plt.plot(l,s,color="black")

    plt.xlabel("Number of counts per 2 seconds")
    plt.ylabel("Number of occurrences (Poisson)")

    # Interpolated probability at x = μ + σ; for marking σ in the graph.
    xx = sp.searchsorted(l,mu_plus_sigma) - 1
    v = ((s[xx+1] -  s[xx])/(l[xx+1]-l[xx])) * (mu_plus_sigma - l[xx])
    v += s[xx]

    ax = plt.gca()
    # Reset axis range and ticks.
    ax.axis([-0.5,10, 0, 40])
    ax.set_xticks(range(1,10), minor=True)
    ax.set_yticks(range(0,41,8))
    ax.set_yticks(range(4,41,8), minor=True)

    # Draw arrow and then place an opaque box with μ in it.
    ax.annotate("", xy=(mu,29), xycoords="data", xytext=(mu, 13),
                textcoords="data", arrowprops=dict(arrowstyle="->",
                                                   connectionstyle="arc3"))
    bbox_props = dict(boxstyle="round", fc="w", ec="w")
    ax.text(mu, 21, r"$\mu$", va="center", ha="center",
            size=15, bbox=bbox_props)

    # Draw arrow and then place an opaque box with σ in it.
    ax.annotate("", xy=(mu,v), xytext=(mu_plus_sigma,v),
                arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
    bbox_props = dict(boxstyle="round", fc="w", ec="w")
    ax.text(mu+(sigma/2.0), v, r"$\sigma$", va="center", ha="center",
            size=15, bbox=bbox_props)

    # Refresh plot and save figure.
    plt.draw()
    plt.savefig("simulate_poisson.png")

def knn():
    from sklearn import preprocessing, cross_validation, neighbors
    path = "breast-cancer-wisconsin.data"
    df = pd.read_csv(path)
    print(df.head())
    
    df = df.replace("?", np.nan).dropna()
    df.drop(["id"], 1, inplace=True)
    
    x = np.array(df.drop(["class"], 1))
    y = np.array(df["class"])
    
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)
    clf = neighbors.KNeighborsClassifier()
    clf.fit(x_train, y_train)
    
    accuracy = clf.score(x_test, y_test)
    print(accuracy)
    
def get_quandl_symbols(symbols):
    start_date, end_date = '2016-01-01', '2017-12-31'
    df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq="D"))
    
    for symbol in symbols:
        quandldata = qdl.get_table("WIKI/PRICES", qopts={"columns":["date", "adj_close"]},
                                   ticker=symbol, date={'gte': start_date, 'lte' : end_date})
        print(quandldata)
        df[symbol] = quandldata
    return df
    
def chapter_3():
    #3.8 -> Simple Linear Regression on Auto data set
    dat = pd.read_csv("Auto.csv")
    dat = dat.replace("?", np.nan).dropna()
    xs, ys = dat["horsepower"].astype(float), dat["mpg"].astype(float)
    results = pd.ols(y=ys, x=xs)
    print(results)
    slope, intercept = results.beta
    r2, f_stat, p_value = results.r2, results.f_stat['f-stat'], results.p_value['intercept']
    fit = [slope*x + intercept for x in xs]
    prediction = results.predict()
    residuals = ys.astype(float) - prediction
    standardized_residuals = (residuals - residuals.mean()) / \
                             (residuals.max() - residuals.min())
    residual_standard_error = results.rmse                       
    percent_error = 100 * residual_standard_error / ys.mean()        

    f = plt.figure()
    ax = f.add_subplot(221)
    ax2 = f.add_subplot(223)
    ax3 = f.add_subplot(222)
    ax4 = f.add_subplot(224)
    ax.scatter(xs, ys, label="r2=%f; f=%f; p=%f" % (
                                                     r2, f_stat, p_value))
    ax.plot(xs, fit, color="r", label="f(x) = %f * x + %f" % (
                                               slope, intercept))
    # non-linear if this plot is not a random scatter plot
    ax2.scatter(xs, residuals, color="r", label="percent error=%f" % percent_error)
    ax2.axhline(0, color="k")
    ax3.scatter(prediction, residuals, label="residual standard error=%f" % residual_standard_error)
    ax3.axhline(0, color="k")
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
    results = pd.ols(y=ys, x=xs)
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
    
    f = plt.figure()
    ax = f.add_subplot(221)
    ax2 = f.add_subplot(223)
    ax3 = f.add_subplot(222)
    ax4 = f.add_subplot(224)
    ax.scatter(results.y_fitted, residuals, label='RSE=%f' % residual_standard_error)
    ax.axhline(0, color="k")
    ax.set_xlabel("y fitted values")
    ax.set_ylabel("residuals")
    ax2.scatter(results.y_fitted, standardized_residuals, label='percent error=%f' % percent_error)
    ax2.axhline(0, color="k")
    ax2.set_xlabel("y fitted values")
    ax2.set_ylabel("standardized residuals")
 
    for _ax in [ax, ax2, ax3, ax4]:
        _ax.legend(loc="best")
    plt.show()
    
if __name__ == "__main__":
    """
    x = 98.24923076
    mean = 98.6
    sigma = 0.064304

    z = (mean - x) / sigma

    p_value = stats.norm.cdf(x, mean, sigma)
    
    binom_pmf()
    norm_cdf()
    simulate_poisson()
    """
    #from sys import argv
    #product = argv[1]
    
    chapter_3()
