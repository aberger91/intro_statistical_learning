import os
import quandl as qdl
import warnings

class QuandlError(Exception):
    errors = {
              0: 'could not register API token',
              1: 'could not load data from quandl'
              }
    def __init__(self, msg, *args):
        if len(args) > 2:
            arg = ' '.join(args[2:])
            self.msg = '%s:\n%s' % (self.errors.get(msg, ''), arg)
        else:
            self.msg = self.errors.get(msg, '')


class Quandl(object):
    '''
    to register your token, set an environment variable QUANDL_API
    '''
    # mapping of product code to database on quandl
    PRODUCT_CODES = {'GC': 'CHRIS/CME_GC1',
                     'CL': 'CHRIS/CME_CL1',
                     'WTI': 'EIA/PET_RWTC_D', # WTI Crude
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
                     'UNEMPLOY': 'FRED/UNEMPLOY',
                     'UNEMPLOY_RATE': 'FRED/UNRATE'
                     }
    API_ENV_VAR = "QUANDL_API"

    def __init__(self):
        self._key = self.API_ENV_VAR
        self.key = self._key

    @property
    def key(self):
        '''
        read-only
        '''
        return self._key

    @key.setter
    def key(self, value):
        if value != self.API_ENV_VAR:
            raise QuandlError(0, value)
        _key = os.getenv(value)
        if _key:
            qdl.ApiConfig.api_key = _key
        else:
            warnings.warn("could not register quandl key with environment variable: %s" % value)
        self._key = _key
            
    def get_data(self, product, start="2014-01-01", end="2018-01-01"):
        """
        if product code no found in dict, thn defaults to WIKI database
        """
        _product = self.PRODUCT_CODES.get(product, '')
        if not _product:
            _product = "WIKI/%s" % product
        try:
            df = qdl.get(_product, start_date=start, end_date=end)
        except Exception as e:
            raise QuandlError(1, product)
        return df
