import matplotlib.pyplot as plt
import numpy as np

def timeseries_comparison(pred, actual, time, len_xtrain, numerical_model = 'ECCO', figsize = (10, 3)):

    """
    Plotting the results for predicted vs. actual timeseries in the same manner as Solodoch et al. (2023) 
    for a temporal split (e.g., first 70% of timeseries is train, last 30% is test) 

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