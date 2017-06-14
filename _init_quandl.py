import quandl as qdl

QUANDL_PRODUCTS = {'GC': 'CHRIS/CME_GC1',
                   'CL': 'CHRIS/CME_CL1',
                   'WTI': 'EIA/PET_RWTC_D',
                   'GDP': "FRED/GDP",
                   'NG': 'CHRIS/CME_NG1',
                   'ZC': 'CHRIS/CME_C1',
                   'ZS': 'CHRIS/CME_S1',
                   'ZL': 'CHRIS/CME_BO1',
                   'HO': 'CHRIS/ICE_O1',
                   '6E': 'CHRIS/CME_EC1',
                   '6C': 'CHRIS/CME_CD1',
                   '6A': 'CHRIS/CME_AD1',
                   '6S': 'CHRIS/CME_SF1',
                   '6B': 'CHRIS/CME_BP1',
                   '6J': 'CHRIS/CME_JY1',
                   'ZN': 'CHRIS/CME_TY1',
                   'SB': 'CHRIS/ICE_SB1',
                   'KC': 'CHRIS/ICE_KC1',
                   'CC': 'CHRIS/ICE_CC1',
                   'ZW': 'CHRIS/CME_W1',
                   'BC': 'BAVERAGE/USD',  #  BitCoin (Last)
                   'FF': 'FRED/DFF',  # FED FUNDS Effective Rate (Value)
                   'ES': 'CHRIS/CME_SP1',
                   'DX': 'CHRIS/ICE_DX1',
                   'PL': 'CHRIS/CME_PL1',
                   'SI': 'CHRIS/CME_SI4',
                   'RY': 'CHRIS/CME_RY1',
                   'VXMT': 'CBOE/VXMT',  # Mid-Term VIX
                   'NROUST': 'FRED/NROUST',
                   'NROU': 'FRED/NROU',
                   'UNEMPLOY': 'FRED/UNEMPLOY',
                   'UNEMPLOY_RATE': 'FRED/UNRATE'
                   }

API_KEY_PATH = "C:/Users/Andrew/api/"

def get_quandl_data(product, 
                    start="2015-01-01", 
                    end="2018-01-01"):
    """
    provides a short-hand lookup for products
    you do not need to know the database, only the product code
    """
    if product not in QUANDL_PRODUCTS:
        '''
        default to the WIKI database for stocks
        '''
        _product = "WIKI/%s" % product
    else:
        _product = QUANDL_PRODUCTS[product]
    df = qdl.get(_product, 
                 start_date=start, 
                 end_date=end)
    return df
