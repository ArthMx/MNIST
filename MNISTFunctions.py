import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time

def digit_plot(digit):
    '''
    Plot a single digit.
    input : (1, 784) array
    '''
    digit_reshaped = digit.values.reshape(28,28)
    
    fig, ax = plt.subplots()
    ax.imshow(digit_reshaped, cmap='gray_r')
    ax.axis('off')
    
def multidigits_plot(digits, size=None, shape=None, secure=True):
    '''
    Plot n digits, max 100 digits if secure=True.
    
    Input : 
    - digits : dataframe of n digits or (n, 784) array.
    - size : integer, or list or tuple of 2 integers, to set size of figure.
    - shape : shape of the axes, then shape=size, except size is set to an integer.
    - secure : if set to True, will raise an error if digits has more than 100 rows.
    '''
    n = len(digits)
    if secure and n>100:
        raise ValueError('Too much digits to plot, make sure there is maximum 100 digits to plot, or set secure to False')
    
    # find the number of rows x, and columns y, for the axes of the plot
    if shape == None:
        for i in range(1, n):
            if (i*(i-1)) <= n:
                x = i
                y = i
                if x*y >= n:
                    break
                x = i
                y = i+1
                if x*y >= n:
                    break   
    else:
        x = shape[0]
        y = shape[1]
        if not isinstance(size, int):
            size = shape
    fig, ax = plt.subplots(x, y)
    ratio = x/y
    
    # set figure size
    if size==None:
        size = 8
        fig.set_size_inches(size/ratio, size*ratio)
    if isinstance(size, int):
        fig.set_size_inches(size/ratio, size*ratio)
    if isinstance(size, (list, tuple)):
        fig.set_size_inches(size[1], size[0])
    
    axes = ax.ravel()

    # plot the digits if digits is dataframe
    if isinstance(digits, pd.core.frame.DataFrame):
        for ax, index in zip(axes, digits.index):
            digit = digits.loc[index, :]
            digit_reshaped = digit.values.reshape(28,28)
            ax.imshow(digit_reshaped, cmap='gray_r')
    
    # plot the digits if digits is a 2D array
    else:
        for ax, digit in zip(axes, digits):
            digit_reshaped = digit.values.reshape(28,28)
            ax.imshow(digit_reshaped, cmap='gray_r')
    
    # hide axis
    for ax in axes:
        ax.axis('off')

def multi_estimators_results(estimators, X_train, X_test, y_train, y_test):
    '''
    input : - estimators :list of estimators
            - X_train, X_test, y_train, y_test : training and test set
    output : tuple : results dataframe and list of estimators fitted
    '''
    cols= ['Train Accuracy Score', 'Test Accuracy Score', 'Fitting Time']
    results_df = pd.DataFrame(columns=cols)

    for estimator in estimators:
        #get estimator name
        estimator_name = estimator.__class__.__name__

        # fit estimator and get time of fit
        t0 = time.time()
        estimator.fit(X_train, y_train)
        t1 = time.time()
        results_df.loc[estimator_name, 'Fitting Time'] = t1 - t0

        # get train accuracy score
        y_train_pred = estimator.predict(X_train)
        train_score = accuracy_score(y_train, y_train_pred)
        results_df.loc[estimator_name, 'Train Accuracy Score'] = train_score

        # get Test accuracy score
        y_test_pred = estimator.predict(X_test)
        test_score = accuracy_score(y_test, y_test_pred)
        results_df.loc[estimator_name, 'Test Accuracy Score'] = test_score

    return results_df, estimators