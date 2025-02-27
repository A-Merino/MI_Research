import numpy as np
import matplotlib.pyplot as plt
import torch

def fmppmi(map1, map2):
    '''
        Function that calculates the paired-pixel mutual
        information (PPMI) of two feature maps 

    ''' 

    # Change range of values to 0 to 1 in maps
    map1 = torch.div(map1, torch.max(map1))
    map2 = torch.div(map2, torch.max(map2))

    # Calculate the marginal PMFs of each image
    Xpmf = fmPMF(map1)
    Ypmf = fmPMF(map2)
    # Calculate the joint PMF of the images
    joint = jointPMF(map1, map2, (list(Xpmf.keys()), list(Ypmf.keys())))

    outtie = []
    prev_MI = {}
    # Go through each pixel in each row of feature map
    for row in map1:
        that = []
        for pixel in row:

            # Round the pixel intensity to get pmf value
            r = np.round(np.float32(pixel), 2)

            # Get probability of pixel happening in map1
            # Send pmf of map2
            # Send joint of map1 and map2 at value of map1
            # Send map2
            if r in prev_MI.keys():
                that.append(prev_MI[r])
            else:
                mi = pixels(Xpmf[r], Ypmf, joint[r], map2)
                prev_MI[r] = mi
                that.append(mi)

        outtie.append(that)
    return outtie


def pixels(prob, pmf, joint, fm):

    # Initialize the sum
    total = 0
    # print(joint)
    # Go through each pixel in feature map
    for row in fm:
        for pixel in row:
            # Get rounded values to recieve pmf
            r = np.round(np.float32(pixel), 2)
            # Add MI to total
            total += calcMI(joint[r], prob, pmf[r])

    # print(total)
    return total


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


def fmPMF(fm):
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
    chan = np.round(np.array(fm).flatten(), 2)
    # count the bins
    val, count = np.unique(chan, return_counts=True)

    # Divide each bin by the amount of
    # to get the pmf pixels
    pmf = count / len(chan)

    # Create a dictionary with each bin as a key
    # and the pmf(x) as the value
    daPMF = {}
    for a, b in zip(val, pmf):
        daPMF[a] = float(b)

    print(daPMF)
    return daPMF


def jointPMF(map1, map2, bins):
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
    fm1 = np.round(np.array(map1).flatten(), 2)
    fm2 = np.round(np.array(map2).flatten(), 2)

    histo = np.histogram2d(fm1, fm2
        , bins=(x_bin, y_bin)  # bins set to 100 for rounding purposes
        )
    plt.imshow(histo[0])
    plt.show()
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
            y_probs[y_e] = float(pixel / allPoints)

        x_probs[edge] = y_probs

    # joint pmf of feature maps is complete
    return x_probs



# print(
    # jointPMF(np.random.andom((300,300)), np.random.random((300,300)))[0.32]
    # fmPMF(np.random.random((300,300)))
    # fmppmi(np.random.random((300,300)), np.random.random((300,300)))
    # )


# b = np.random.random((300,300))
# g = np.random.random((300,300))

# jointPMF(b, g)
# jointPMF(g, b)

