    ####################################################################################################################
import pandas as pd
import numpy as np
import scipy.optimize as so

    ####################################################################################################################

def plot_various_efficient_frontier(n_plot_points, expected_returns, 
                                    expected_returns_covariance_matrix, style = '.-', legend = True, 
                                    show_capital_market_line = False, riskfree_rate = 0, show_equally_weighted = False, 
                                    show_global_minimum_variance = False):
    """
    Plots the multi-asset efficient frontier returns and various other methods:

    1- Adjust your data series of returns before the code running.

    2- Adjust the number of points that you want to plot in main function calling, I usually use 20 - 30 points,
       but it is something that you can choose in "n_plot_points".

    3- Define "expected_returns" before calling main function, it can be a part of your return series or exactly
       all the return series. Be carefull about that you must first annualize your returns using
       "annualized_rets.py", then you can call the main function.

    4- Define "expected_returns_covariance_matrix" just by using simple covariance formula below:
       "expected_returns_covariance_matrix = expected_returns.cov()", you can define another function for that
       seperately as an exercise.

    5- Defined "portfolio_return" function to compute the return on a portfolio from returns and constituent weights.

    6- Defined "portfolio_volatility" function to compute the volatility of a portfolio from a covariance
       matrix and constituent weights.

    7- Defined "minimize_volatility" function to compute the optimal weights that achieve the target return
       given a set of expected returns and a covariance matrix.

    8- Defined "optimal_weights" function to compute a list of weights that represent a grid of n_plot_points
       on the efficient frontier returns

    9- Part one of main function: plot efficient frontier.

    10- Change "False" to "True" one by one to use other methods.

    11- Defined "weights_of_maximum_sharp_ratio" function to compute the weights of the portfolio that gives
       you the maximum sharpe ratio given the riskfree rate and expected returns and a covariance matrix.

    12- Defined "negative_sharp_ratio" inside the "weights_of_maximum_sharp_ratio" function to returns the
       negative of the sharp ratio of the given portfolio.

    13- If condition to "show_capital_market_line" to use part two of main function:
       plot efficient frontier with capital market line using risk free rate in measuring return
    
    14- If condition to "show_equally_weighted" to use part three of main function:
       plot efficient frontier, capital market line and equally weighted (EW) condition.

    15- Defined "global_minimum_variance" function to returns the weights of the Global Minimum Variance
        portfolio given a covariance matrix to improve the lack of robustness of Markowits analysis.

    16- If condition to "show_global_minimum_variance" to use part four of main function:
       plot efficient frontier, capital market line, EW and GMW points.
    """

    ####################################################################################################################

    def portfolio_return(weights, returns):
        # Weights and returns are a numpy array or Nx1 matrix
        return weights.T @ returns  

    def portfolio_volatility(weights, covmat):
        # Weights are a numpy array or N x 1 maxtrix and covariance matrix is an N x N matrix
        return (weights.T @ covmat @ weights) ** 0.5  

    def minimize_volatility(target_return, expected_returns, expected_returns_covariance_matrix):
        n = expected_returns.shape[0]
        initial_guess = np.repeat(1 / n, n)
        # Creates N-tuple of 2-tuples
        bounds = ((0.0, 1.0),) * n 
        # Construct the constraints
        weights_sum_to_1 = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
        return_is_target = {'type': 'eq',
                            'args': (expected_returns,),
                            'fun': lambda weights, expected_returns: 
                                   target_return - portfolio_return(weights, expected_returns)}
        weights = so.minimize(portfolio_volatility, initial_guess,
                              args = (expected_returns_covariance_matrix,), method='SLSQP',
                              options = {'disp': False},
                              constraints = (weights_sum_to_1,return_is_target),
                              bounds = bounds)
        return weights.x

    def optimal_weights(n_plot_points, expected_returns, expected_returns_covariance_matrix):
        target_rs = np.linspace(expected_returns.min(), expected_returns.max(), n_plot_points)
        weights = [minimize_volatility(target_return, expected_returns, 
                   expected_returns_covariance_matrix) for target_return in target_rs]
        return weights

    ############################## Part one of main function: plot efficient frontier ##################################

    weights = optimal_weights(n_plot_points, expected_returns, expected_returns_covariance_matrix)
    rets = [portfolio_return(w, expected_returns) for w in weights]
    vols = [portfolio_volatility(w, expected_returns_covariance_matrix) for w in weights]
    efficient_frontier = pd.DataFrame({"Returns": rets, "Volatility": vols})
    ax = efficient_frontier.plot.line(x = "Volatility", y = "Returns", style = style, legend = legend)

    ####################################################################################################################

    def weights_of_maximum_sharp_ratio(riskfree_rate, expected_returns, expected_returns_covariance_matrix):
        n = expected_returns.shape[0]
        initial_guess = np.repeat(1 / n, n)
        # Creates N-tuple of 2-tuples
        bounds = ((0.0, 1.0),) * n 
        # Construct the constraints
        weights_sum_to_1 = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
        def negative_sharp_ratio(weights, riskfree_rate, expected_returns, expected_returns_covariance_matrix):
            returns = portfolio_return(weights, expected_returns)
            volatility = portfolio_volatility(weights, expected_returns_covariance_matrix)
            return -(returns - riskfree_rate) / volatility
    
        weights = so.minimize(negative_sharp_ratio, initial_guess, bounds = bounds, method = 'SLSQP', 
                              args = (riskfree_rate, expected_returns, expected_returns_covariance_matrix), 
                              options = {'disp': False}, constraints = (weights_sum_to_1,))
        return weights.x
    ################## Part two of main function: plot efficient frontier with capital market line #####################

    if show_capital_market_line:
        ax.set_xlim(left = 0)
        w_weights_of_maximum_sharp_ratio = weights_of_maximum_sharp_ratio(riskfree_rate, 
                                                                          expected_returns, 
                                                                          expected_returns_covariance_matrix)
        r_weights_of_maximum_sharp_ratio = portfolio_return(w_weights_of_maximum_sharp_ratio, expected_returns)
        vol_weights_of_maximum_sharp_ratio = portfolio_volatility(w_weights_of_maximum_sharp_ratio, 
                                                                  expected_returns_covariance_matrix)
        capital_market_line_x = [0, vol_weights_of_maximum_sharp_ratio]
        capital_market_line_y = [riskfree_rate, r_weights_of_maximum_sharp_ratio]
        ax.plot(capital_market_line_x, capital_market_line_y, color = 'green', marker = 'o',
        linestyle = 'dashed', linewidth = 2, markersize = 10)

    ### Part three of main function: plot efficient frontier, capital market line and equally weighted (EW) condition ##

    if show_equally_weighted:
        n = expected_returns.shape[0]
        weight_equally_weighted = np.repeat(1/n, n)
        return_equally_weighted = portfolio_return(weight_equally_weighted, expected_returns)
        volatility_equally_weighted = portfolio_volatility(weight_equally_weighted, expected_returns_covariance_matrix)
        ax.plot([volatility_equally_weighted], [return_equally_weighted], color='gold', marker='o', markersize=10)

    ####################################################################################################################

    def global_minimum_variance(expected_returns_covariance_matrix):
        n = expected_returns_covariance_matrix.shape[0]
        return weights_of_maximum_sharp_ratio(0, np.repeat(1, n), expected_returns_covariance_matrix)
    
    ############ Part four of main function: plot efficient frontier, capital market line, EW and GMV points ###########

    if show_global_minimum_variance:
        weight_global_minimum_variance = global_minimum_variance(expected_returns_covariance_matrix)
        return_global_minimum_variance = portfolio_return(weight_global_minimum_variance, expected_returns)
        volatility_global_minimum_variance = portfolio_volatility(weight_global_minimum_variance, 
                                                           expected_returns_covariance_matrix)
        ax.plot([volatility_global_minimum_variance], [return_global_minimum_variance], 
                color = 'blue', marker = 'o', markersize = 10)
        return ax

    #################################################################################################################### 
