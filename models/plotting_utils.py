import matplotlib.pyplot as plt
import numpy as np

def timeseries_comparison(pred, actual, time, len_xtrain, numerical_model = 'ECCO', figsize = (10, 3)):

    """
    Plotting the results for predicted vs. actual timeseries in the same manner as Solodoch et al. (2023) 
    for a temporal split (e.g., first 70% of timeseries is train, last 30% is test).

    Parameters
    ----------
    pred : numpy.array
        array of MOC predictions on the full dataset
    actual : numpy.array
        array of all observed MOC model values
    time : numpy.array
        array of datetime timestamps for each data point
    len_xtrain : integer
        the length of the training set
    numerical_model : string
        the numerical model used, either ECCO or ACCESS
    figsize : tuple
        the size of the figure

    Returns
    -------
    fig, ax : matplotlib.figure, matplotlib.axes
        figure and axis objects from matplotlib
    """

    # Set up figure and axes to return later
    fig, ax = plt.subplots(figsize = figsize)

    ax.plot(time, actual, label = numerical_model)
    ax.plot(time, pred, label = 'Predicted')

    #  shading the train period
    y_lower, y_upper = plt.gca().get_ylim()
    x_pos = time[ : len_xtrain]

    ax.fill_between(x = x_pos, 
                    y1 = np.repeat(y_lower, len(x_pos)), 
                    y2 = np.repeat(y_upper, len(x_pos)),
                    alpha = 0.2, 
                    color = 'gray')

    ax.margins(x = 0, y = 0)

    #  adding legend
    ax.legend(loc = 'lower right', edgecolor = 'black', framealpha = 1)

    #  labeling
    ax.set_ylabel('MOC Strength [Sv]', weight = 'bold')

    return fig, ax

def pred_vs_actual(train_pred, test_pred, train_label, test_label, numerical_model = 'ECCO', figsize = (4, 4)):

    """
    Plotting the results for predicted vs. actual MOC predictions as a scatterplot.

    Parameters
    ----------
    train_pred : numpy.array
        array of MOC predictions on the train set
    test_pred : numpy.array
        array of MOC predictions on the test set
    train_label : numpy.array
        array of MOC model values for the train set
    test_pred : numpy.array
        array of MOC model values for the test set
    numerical_model : string
        the numerical model used, either ECCO or ACCESS
    figsize : tuple
        the size of the figure

    Returns
    -------
    fig, ax : matplotlib.figure, matplotlib.axes
        figure and axis objects from matplotlib
    """

    # Set up figure and axes to return later
    fig, ax = plt.subplots(figsize = figsize)

    #  computing (linear) correlation for train and test set
    train_corr = round(np.corrcoef(train_pred, train_label)[0, 1], 3)
    test_corr = round(np.corrcoef(test_pred, test_label)[0, 1], 3)

    #  adding 1:1 line
    ax.axline((0, 0), slope = 1, color = 'black', linestyle = '--', zorder = 0)

    #  adding scatterplot for train and test set separately
    ax.scatter(train_pred, train_label, alpha = 0.4, color = 'blue', s = 15, label = f'Train (Cor.={train_corr})', zorder = 20)
    ax.scatter(test_pred, test_label, alpha = 0.4, color = 'orange', s = 15, label = f'Test (Cor.={test_corr})', zorder = 20)

    #  adding legend
    ax.legend(loc = 'upper left', edgecolor = 'black', framealpha = 1)

    #  ensuring the axes limits are the same and adding grid
    ax.axis('square')
    ax.grid(alpha =  0.3)

    #  labeling axes
    ax.set_xlabel('Predicted MOC Strength [Sv]', weight = 'bold')
    ax.set_ylabel(f'{numerical_model} MOC Strength [Sv]', weight = 'bold')

    return fig, ax