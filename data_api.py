import os

from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt



ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY')

ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
data, meta_data = ts.get_intraday(symbol='MSFT',interval='1min', outputsize='full')

data['4. close'].plot()
plt.title('Intraday Times Series for the MSFT stock (1 min)')
plt.show()
