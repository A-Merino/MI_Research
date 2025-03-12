import numpy as np
import matplotlib.pyplot as plt
import torch

def streammi(stream1, stream2):
    '''
        Function that calculates the mutual information
        between two streams of data

    ''' 

    # Change range of values to 0 to 1 in maps
    stream1 = torch.div(stream1, torch.max(stream1))
    stream2 = torch.div(stream2, torch.max(stream2))

    # Calculate the marginal PMFs of each image
    Xpmf = linePMF(stream1)
    Ypmf = linePMF(stream2)

    # Calculate the joint PMF of the images
    joint = jointPMF(stream1, stream2, (list(Xpmf.keys()), list(Ypmf.keys())))

    mi = 0
    # Go through each pixel in each row of feature map
    for p1, p2 in zip(stream1, stream2):


        # Round the pixel intensities to get pmf value
        r1 = float(torch.round(p1, decimals=2))
        r2 = float(torch.round(p2, decimals=2))

        # Calculate mutual information between two pixels
        mi += calcMI(joint[r1][r2], Xpmf[r1], Ypmf[r2]) 

    return float(mi)


def calcMI(pxy: float, px: float, py: float) -> float:
    '''
        Calculates the mutual information between
        two variables x and y

        Parameters:
            - pxy: The probability P(X=x, Y=y), which is the
            joint probability that x and y both occur
            - px: The probability P(X=x), which is the marginal
            probablity that x occurs
            - py: The probability P(Y=y), which is the marginal
            probablity that y occurs

        Output:
            The calculated result from the formula of Mutual Information:

                    P(X=x, Y=y) * log(P(X=x, Y=y) / P(X=x) * P(Y=y))

            Where the log is representative of the natural log, ln

            If the calculation inside the log is less than 1, we will return 0
            since information cannot be negative
    '''

    assert px > 0
    assert py > 0

    inLog = pxy / (px * py)

    if inLog < 1:
        return 0

    return pxy * np.log(inLog)


def linePMF(fm):
    '''
        Function which calculates the pmf of a feature
        map

        Parameters
        ----------

        fm : array_like
            The intensity values of a feature map

        Returns
        -------
        PMF : dictionary
            A dictionary which hold the marginal
            pmf of the feature map.

            The maximum size of the output is len(PMF) = 100
            since we round to the hundreths to get actual
            probabilties
    '''

    # Here we round the intensity values to the thousandths
    # to get relatively larger bins for calculating the pmf
    chan = torch.round(fm, decimals=2)
    # count the bins
    val, count = torch.unique(chan, return_counts=True)

    # Divide each bin by the amount of
    # to get the pmf pixels
    pmf = torch.div(count, len(chan))

    # Create a dictionary with each bin as a key
    # and the pmf(x) as the value
    daPMF = {}
    for a, b in zip(val, pmf):
        daPMF[float(a)] = float(b)

    return daPMF


def jointPMF(s1, s2, bins):
    '''
        Function which calculates the joint PMF of
        two feature maps

        Parameters
        ----------
            map1: The intensity values of a feature map
            map2: The intensity values of a feature map

        Output:
            The joint pmf of the intensity values
            of the two input maps

            It is technically a list of dictionaries,
            but is essentially the same as an array
            of size (100 x 100)
    '''

    x_bin, y_bin = bins

    x_bin.append(x_bin[-1] + 0.1)
    y_bin.append(y_bin[-1] + 0.1)

    # Round values to thousandths for bins
    fm1 = torch.round(s1, decimals=2)
    fm2 = torch.round(s2, decimals=2)

    histo = np.histogram2d(fm1, fm2
        , bins=(x_bin, y_bin)  # bins set to 100 for rounding purposes
        )

    # This will give us a dictionary of y probabilities
    x_probs = {}
    allPoints = sum(sum(histo[0]))


    # Go through each bin from histogram indexed by x probabilities
    for edge, row in zip(bins[0], histo[0]):
        y_probs = {}

        # Go through each x probability for the
        # y probability we are looking at
        for y_e, pixel in zip(bins[1], row):
            # Calculate P(X=x, Y=y)
            y_probs[float(y_e)] = float(pixel / allPoints)

        x_probs[float(edge)] = y_probs

    # joint pmf of feature maps is complete
    return x_probs

