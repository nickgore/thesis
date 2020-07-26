"""
Backtesting.
"""
import numpy as np
import pandas as pd
from scipy import stats



def test_outcome(p_value, statistic):
    return 'Accept' if statistic > p_value else 'Reject'


class BackTest:
    def __init__(self, actual, forecast, alpha):
        self.actual = actual
        self.forecast = forecast
        self.alpha = alpha

    def hit_series(self):
        return np.array((self.actual < self.forecast) * 1)

    def back_test(self):
        hits = self.hit_series()   # hits
        transitions = hits[1:] - hits[:-1]  # transitions

        # transitions: nij denotes state i is followed by state j nij times
        n01 = (transitions == 1).sum()
        n10 = (transitions == -1).sum()
        n11 = (hits[1:][transitions == 0] == 1).sum()
        n00 = (hits[1:][transitions == 0] == 0).sum()

        # Probabilities of the transitions from one state to another
        p01 = n01 / (n00 + n01)
        p11 = n11 / (n11 + n10)
        p = (n10 + n11) / (n01 + n00 + n10 + n11)

        # Unconditional Coverage (Kupiec's POF test)
        if n10 + n11 > 0:
            uc_h0 = (n01 + n00) * np.log(1 - self.alpha) + (n10 + n11) * np.log(self.alpha)
            uc_h1 = (n01 + n00) * np.log(1 - p) + (n10 + n11) * np.log(p)
            lr_pof = -2 * (uc_h0 - uc_h1)

            # Independence (Christoffersen’s independence test)
            ind_h0 = (n00 + n01) * np.log(1 - p) + (n01 + n11) * np.log(p)
            ind_h1 = n00 * np.log(1 - p01) + n01 * np.log(p01) + n10 * np.log(1 - p11)
            if p11 > 0:
                ind_h1 += n11 * np.log(p11)
            lr_ind = -2 * (ind_h0 - ind_h1)

            # Conditional coverage Joint test of unconditional coverage and independence
            # (Christoffersen’s Interval Forecast Test)
            lr_cc = lr_pof + lr_ind

            # Stack results
            statistics = pd.concat(
                [
                    pd.Series([lr_pof, lr_ind, lr_cc]),
                    pd.Series(
                        [
                            1 - stats.chi2.cdf(lr_pof, 1),
                            1 - stats.chi2.cdf(lr_ind, 1),
                            1 - stats.chi2.cdf(lr_cc, 2)
                        ]
                    ),
                    pd.Series(
                        [
                            test_outcome(1 - stats.chi2.cdf(lr_pof, 1), lr_pof),
                            test_outcome(1 - stats.chi2.cdf(lr_ind, 1), lr_ind),
                            test_outcome(1 - stats.chi2.cdf(lr_cc, 2), lr_cc)
                        ])
                ],
                axis = 1
            )
        else:
            statistics = pd.DataFrame(np.zeros((3, 3))).replace(0, np.nan)

        # formatting
        statistics.columns = ['statistic', 'p-value', 'test outcome']
        statistics.index = ['Unconditional', 'Independence', 'Conditional']

        return statistics


