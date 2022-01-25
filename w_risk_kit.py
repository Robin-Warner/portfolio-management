import pandas as pd
import numpy as np
import scipy.stats
from scipy.optimize import minimize

def drawdown(return_series: pd.Series):
    '''
    Takes a pandas time series of asset returns
    Computes and returns a DataFrame that Contains:
        the wealth index
        the previous peaks
        percent drawdowns
    '''
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({'Wealth': wealth_index,
                        'Peaks' : previous_peaks,
                        'Drawdown' : drawdowns})

def calc_mth_returns(r: pd.Series):
    return (1+r).groupby(pd.Grouper(freq='M')).prod()-1

def semideviation(r):
    '''
    Returns the semideviation (aka negative standard deviation) of r
    r must be a Series or a DataFrame
    '''
    is_negative = r < 0
    return r[is_negative].std(ddof=0)

def skewness(r):
    '''
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or Dataframe
    Returns a Float or a Series
    '''
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    '''
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or Dataframe
    Returns a Float or a Series
    '''
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def is_normal(r, level=0.01):
    '''
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    '''
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level

def var_historic(r, level=5):
    '''
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns 
    fall below that number, and the 100-"level" percent are above
    '''
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else: 
        raise TypeError('Expected r to be Series or DataFrame')
        
def var_gaussian(r, level=5, modified=False):
    '''
    Returns the parametric Gaussian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    '''
    # compute the Z score assuming it was Gaussian
    z = scipy.stats.norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6+
                (z**3 - 3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )    
    return -(r.mean() + z*r.std(ddof=0))

def cvar_historic(r, level=5):
    '''
    Computes the Conditional VaR of Series or DataFrame
    '''
    if isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    elif isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    else: 
        raise TypeError('Expected r to be Series or DataFrame')

def annualize_rets(r, periods_per_year):
    '''
    Annualizes a set of returns
    We should infer the periods per year
    but that is currently left as an excersise to the reader
    '''
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r, periods_per_year):
    '''
    Annualizes the vol of a set of returns
    We should infer the periods per year
    but that is currently left as an excersise to the reader
    '''
    return r.std()*(periods_per_year**0.5)

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    '''
    Computes the annualized sharpe ratio of a set of returns
    '''
    #convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol

def portfolio_return(weights, returns):
    '''
    weights -> returns
    '''
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    '''
    weights -> vol
    '''
    return (weights.T @ covmat @ weights)**0.5

def minimize_vol(target_return, er, cov):
    '''
    target return -> weight vector
    '''
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1 
    }
    results = minimize(portfolio_vol, init_guess,args=(cov,), 
                       method='SLSQP', 
                       options={'disp': False},
                       constraints=(return_is_target, weights_sum_to_1),
                       bounds=bounds                        
                      )
    return results.x

def optimal_weights(n_points, er, cov):
    '''
    list of returns to run the optimizer to minimize the vol and give back weights
    '''
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def msr(riskfree_rate, er, cov):
    '''
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the risk free rate, expected returns and a covariance matrix
    '''
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),)*n
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1 
    }
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        '''
        Returns the negative of the sharpe ratio, given weights
        '''
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    results = minimize(neg_sharpe_ratio, init_guess,
                       args=(riskfree_rate, er, cov,), method='SLSQP', 
                       options={'disp': False},
                       constraints=(weights_sum_to_1),
                       bounds=bounds                        
                      )
    return results.x

def plot_ef(n_points, er, cov, style='.-', show_cml=False, riskfree_rate=0, figsize=(12,6)):
    '''
    Plots the N-asset efficient frontier
    '''
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        'Efficient Frontier': rets,
        'Volatility': vols
    })
    ax = ef.plot.line(x='Volatility', y='Efficient Frontier', style=style, figsize=figsize)
    
    if show_cml:
        ax.set_xlim(left = 0)
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        # Add CML
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, label='CML', color='red', marker='o', linestyle='dashed', markersize = 3, linewidth=2)
        ax.legend(loc='upper right')
        
    return ax

def std_distr_analysis(r, periods_per_year, riskfree_rate):
    '''
    Returns DataFrame with CAIA Standardized Returns Analysis Framework
    '''
    tbl =pd.DataFrame({'Annualized Return':annualize_rets(r,periods_per_year),
              'Annualized Volatility':annualize_vol(r, periods_per_year),
              'Skewness':skewness(r),
              'Kurtosis':kurtosis(r),
              'Is Normal':r.aggregate(is_normal),
              'Sharpe Ratio':sharpe_ratio(r, riskfree_rate, periods_per_year),
              'Gaussian VaR (95%)':var_gaussian(r),
              'Cornish-Fisher VaR (95%)':var_gaussian(r, modified=True),         
              'Monthly Historic VaR (95%)':var_historic(r),
              'Monthly Historic CVaR (95%)':cvar_historic(r),
              'Maximum':r.max(),
              'Maximum Date':r.idxmax(),
              'Minimum':r.min(),
              'Minimum Date':r.idxmin(),
              'Max Drawdown':r.aggregate(lambda col: drawdown(col)['Drawdown'].min()),
              'Max Drawdown Period End':r.aggregate(lambda col: drawdown(col)['Drawdown'].idxmin())
             }).T
        
    return tbl

def trend_info(levels: pd.Series):
    '''
    Takes a pandas time series of asset levels
    Computes and returns a DataFrame that Contains:
        the original level
        returns
        10-day MAVG
        45-day MAVG
        the difference between the 10-day and 45-day MAVG
        the dates when the MAVGs cross (-1: means 10D crosses 45D from above, 1: means 10D crosses 45D from below)
        the trade signal (currently simply a function of the MAVG cross)
    '''
    orig_lvl = levels
    rets = orig_lvl.pct_change().fillna(0)
    mavg_10D = orig_lvl.rolling(10).mean()
    mavg_45D = orig_lvl.rolling(45).mean()
    mavgdiff_10Dminus45D = mavg_10D - mavg_45D
    mavgcross_10Dminus45D = np.sign(np.sign(mavgdiff_10Dminus45D).diff().fillna(0))
    signal = mavgcross_10Dminus45D.replace({0:np.NaN}).fillna(method='ffill').fillna(0)
    strat_return = signal * rets
    return pd.DataFrame({'Asset Level': orig_lvl,
                        'Asset Returns': rets,
                        '10-day SMA' : mavg_10D,
                        '45-day SMA' : mavg_45D,
                        'SMA Diff (10D - 45D)':mavgdiff_10Dminus45D,
                        'SMA Cross (10D - 45D)':mavgcross_10Dminus45D,
                        'Trade Signal': signal,
                         'Strategy Returns': strat_return
                        })

def plot_multi(data, cols=None, spacing=.1, **kwargs):

    from pandas.plotting._matplotlib.style import get_standard_colors

    # Get default color style from pandas - can be changed to any other color list
    if cols is None: cols = data.columns
    if len(cols) == 0: return
    colors = get_standard_colors(num_colors=len(cols))

    # First axis
    ax = data.loc[:, cols[0]].plot(label=cols[0], color=colors[0], **kwargs)
    ax.set_ylabel(ylabel=cols[0])
    lines, labels = ax.get_legend_handles_labels()

    for n in range(1, len(cols)):
        # Multiple y-axes
        ax_new = ax.twinx()
        ax_new.spines['right'].set_position(('axes', 1 + spacing * (n - 1)))
        data.loc[:, cols[n]].plot(ax=ax_new, label=cols[n], color=colors[n % len(colors)], **kwargs)
        ax_new.set_ylabel(ylabel=cols[n])
        
        # Proper legend position
        line, label = ax_new.get_legend_handles_labels()
        lines += line
        labels += label

    ax.legend(lines, labels, loc=0, prop={'size':18})
    return ax