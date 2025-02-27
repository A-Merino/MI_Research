import numpy as np
import matplotlib.pyplot as plt


def colorPPMI(image1, image2):
    '''
        Function which calculates the Paired-Pixel Mutual Information (PPMI)
        of each pixel value between two images

        Parameters:
            - image1: The pixel intensity values of image1
            - image2: The pixel intensity values of image2

        Output:
            None rn it'll take 7 * image_width * image)height seconds to run
                - 7 is about how long it takes, in seconds, to receive the
                mutual information value of 1 pixel to the entire other image

    '''

    # Calculate the marginal PMFs of each image
    Xpmf = imagePMF(image1)
    Ypmf = imagePMF(image2)

    # Calculate the joint PMF of the images
    joint = jointPMF(image1, image2)

    # Grab the pmf for X, joint pmf of x and y, and the
    # pixel values of the channel
    for pmf, jj, channel in zip(Xpmf, joint, image1):

        # Go through each pixel in the channel
        for row in channel:
            for pixel in row:
                # Round the pixel intensity to get pmf value
                r = float(np.round(pixel, 2))
                # Run the PPMI sum
                pixels(pmf[r], Ypmf, jj[r], image2)


def pixels(prob, pmf, joint, image):

    # Initialize the sum
    total = 0

    # Go through pmf of y and channel of image
    for ys, channel in zip(pmf, image):

        # Go through each pixel in the channel
        for row in channel:
            for pixel in row:
                # Get rounded values to recieve pmf
                r = float(np.round(pixel, 2))

                # Round 1 down since it is not in pmf
                if r == 1:
                    r = 0.99

                # Add MI to total
                total += calcMI(joint[r], prob, ys[r])
    print(total)


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

    inLog = pxy / (px * py)

    if inLog < 1:
        return 0

    return pxy * np.log(inLog)


def imagePMF(image):
    '''
        Function which calculates the pmf of each color
        channel in an image

        Parameters
        ----------

        image : array_like
            The intensity values of the image

        Returns
        -------
        rgb : List of dictionaries
            A list of dictionaries which hold the marginal
            pmf of a channel of the input image.

            The maximum size of the output is 3*100
                - 3 being amount of channels
                - 100 is since we round to the hundredths
    '''

    # Initialize array to hold pmfs
    rgb = []

    # Go through each color channel in the image
    for channel in image:

        # Flatten the color channel into a 1D-array.
        # Here we round the intensity values to the thousandths
        # to get relatively larger bins for calculating the pmf
        chan = np.round(np.array(channel).flatten(), 2)

        # count the bins
        val, count = np.unique(chan, return_counts=True)

        # Divide each bin by the amount of
        # to get the pmf pixels
        pmf = count / len(chan)

        # Create a dictionary with each bin as a key
        # and the pmf(x) as the value
        daPMF = {}
        for a, b in zip(val, pmf):
            daPMF[float(a)] = float(b)

        rgb.append(daPMF)

    return rgb


def jointPMF(image1, image2):
    '''
        Function which calculates the joint PMF of
        two images

        Parameters:
            - image1: The rgb values of an image
            - image2: The rgb values of an image

        Output:
            The joint pmf of the intensity values
            of each channel of the two input images

            It is technically a list of dictionaries,
            but is essentially the same as an array
            of size (3 x 100 x 100)
    '''

    # Initialize color channel array
    rgb = []

    # Grab the color channel of both images
    for c1, c2 in zip(image1, image2):

        # Flatten the channels to compute the joint pmf
        # Also round values to thousandths for bins
        fc1 = np.round(np.array(c1).flatten(), 3)
        fc2 = np.round(np.array(c2).flatten(), 3)

        histo = np.histogram2d(fc1, fc2
            , bins=100  # bins set to 100 for rounding purposes
            )

        # This will give us a dictionary of y probabilities
        x_probs = {}
        allPoints = sum(sum(histo[0]))

        # Go through each bin from histogram indexed by y probabilities
        for edge, row in zip(histo[1], histo[0]):
            y_probs = {}

            # Go through each x probability for the
            # y probability we are looking at
            for y_e, pixel in zip(histo[1], row):
                # Calculate P(X=x, Y=y)
                y_probs[float(np.round(y_e, 2))] = float(pixel / allPoints)

            x_probs[float(np.round(edge, 2))] = y_probs

        # joint pmf of color channel is complete
        rgb.append(x_probs)

    return rgb

