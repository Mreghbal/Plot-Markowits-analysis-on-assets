# Plot-Markowits-analysis-on-assets
Plot the multi-asset efficient frontier returns and various other methods

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
