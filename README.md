### Recurrent Neural Networks use for Value
### at Risk estimation. Application to the
### Visegrád Four stock market indexes.


#### Content:
* **Recurrent_Neural_Networks_use_for_Value_at_Risk_estimation.pdf** - current version of work-in-progress thesis

* presentation_notebook.ipynb - notebook for training models on index data, performing forecast and back-testing
* Histogram and Statistics.ipynb - notebook to obtain statistics from data
* backtesting.py - contains definitions of backtest
* LSTM.py - LSTM model
* helpers.py - helper functions

#### Tools:
**LSTM** is deep learning algorithm developed by Hochreiter and
Schmindhuber (1997).
#### Data:
Daily time series for BUX, SAX, PX, WIG20

------
### TO DO LIST

#### Prerequisits:
The requisites are defined in requirements.txt
```
pip3 install -r requirements.txt
```

#### Authors:
Nikanor Goreglyad (40038969)
#### References:
Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long Short-Term Memory. Neural Comput. 9, 8 (November 1997), 
1735–1780. DOI:https://doi.org/10.1162/neco.1997.9.8.1735
