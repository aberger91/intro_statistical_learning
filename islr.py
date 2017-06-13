from quandl import ApiConfig
from matplotlib import style as mpl_style
mpl_style.use(["ggplot"])
from _init_quandl import API_KEY_PATH

try:
    ApiConfig.api_key = open(API_KEY_PATH + "quandl.txt", "r").read()
except Exception as e:
    print("could not register quandl key\n%s" % e)
    
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
