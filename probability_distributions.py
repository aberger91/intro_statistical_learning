import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import math

def normz(val=0.95):
    two_tailed_ztest_critical_value = scipy.stats.norm.ppf((1+val)/2)
    return two_tailed_ztest_critical_value
 
def logit_function(L, k, x0, x):
    '''
    L -> maximum value
    k -> steepness
    x0 -> x-value of sigmoid midpoint
    '''
    f = L / (1 + math.exp(-k * (x - x0)))
    return f

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
    plt.title("PMF for Gaussian of mean=%f & std=%f" % (mean, std))
    plt.show()
    
def norm_cdf(n=100, mean=0.0, std=1.0):
    x = np.linspace(-n/10, n/10, n)
    y = stats.norm.cdf(x, loc=mean, scale=std)
    plt.plot(x, y)
    plt.xlabel("Variate")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF for Gaussian of mean=%f & std=%f" % (mean, std))
    plt.show()
    
def main():
    #binom_pmf()
    #binom_cdf()
    #norm_pmf()
    #norm_cdf()
    L, k, x0 = 1, 1, 0
    xs = range(-10, 11)
    ys = list(map(lambda x: logit_function(L, k, x0, x), xs))
    plt.plot(xs, ys)
    plt.show()
    
if __name__ == "__main__":
    main()