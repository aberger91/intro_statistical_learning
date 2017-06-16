import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime as dt
from _init_quandl import get_quandl_data
import scipy.stats as stats

def calculate_daily_value_at_risk(P, prob, mean, sigma, days_per_year=252.):
    min_ret = stats.norm.ppf(1-prob, 
                             mean/days_per_year, 
                             sigma/np.sqrt(days_per_year))
    return P - P * (min_ret + 1)

def fedfunds():
    start, end = "1970-01-01", "2018-01-01"

    es = get_quandl_data("ES",
                         start=start, 
                         end=end)
    ff = get_quandl_data("FF", 
                         start=start, 
                         end=end)
    unemploy = get_quandl_data("UNEMPLOY",
                               start=start,
                               end=end)                                  
    gdp = get_quandl_data("GDP",
                          start=start,
                          end=end)
    features = pd.Series({"FF": ff, "UNEMPLOY": unemploy, "GDP": gdp})

                                              
    sample_xs = ff.ix[ff.index > dt.datetime(2008, 1, 1)].diff().dropna()
    sample_ys = es.ix[es.index > dt.datetime(2008, 1, 1)].diff().dropna()

    sample_xs = sample_xs.ix[sample_ys.index]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(sample_xs["Value"], sample_ys["Settle"])

    for ix, x, y in list(zip(sample_xs.index, 
                             sample_xs["Value"], 
                             sample_ys["Settle"])):
        ax.annotate(ix, xy=(x, y))
        
    model = sm.OLS(sample_ys["Settle"], sample_xs["Value"])
    fit = model.fit()
    print(fit.summary())

    plt.show()

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

def n_neighbors(n=15):
    from sklearn import neighbors
    from sklearn import datasets
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
    
def fx_regression(pairs=["6A", "6E", "6B", "6J", "6S", "ZN", "ES"]):
    from quandl import ApiConfig
    from _init_quandl import API_KEY_PATH
    try:
        ApiConfig.api_key = open(API_KEY_PATH + "quandl.txt", "r").read()
    except Exception as e:
        print("could not register quandl key\n%s" % e)
    dat = {p:get_quandl_data(p) for p in pairs}
    return dat
    
if __name__ == "__main__":
    fx_regression()
