import numpy as np
import matplotlib.pyplot as plt



def mppmi(image1, image2):
    '''
        My Paired Pixel Mutual Information

        Parameters:
            - image1: The image we will iterate over
            - image2: The image we are comparing image1 to
    '''

    im1pmf = imageMPMF(image1) 


    # for color1, color2 in zip(image1, image2):
    #     for row in color1:
            
    #         colorPPMI(row, color2)

            # for pixel in row:
            #     colorPPMI(pixel, color2)


def colorPPMI(pixel, channel):

    # Do MI pixelwise
    mi = 0

    # Iterate over every pixel
    for row in channel:
        pd = np.histogram2d(pixel, row, bins=[0,1,2,3])[0]
        print(pd)
        plt.hist(pd)
        plt.show()
        # for cell in row:
        #     pd = np.histogram2d([pixel], [cell])
        #     print(pd)
        #     # Calculate MI between two pixels


def imageMPMF(image):
    for channel in image:
        chan = np.array(channel).flatten() 
        print(chan)
        pd = np.histogram(chan
            # , bins=sorted(chan)
            )[0]
        print(pd)
        for g in pd:
            g /= sum(pd)
        print(pd)
        plt.hist(pd, bins='auto', density='True')
        plt.show()




image1 = [[[1,2],[3,4]],
 [[5,6],[7,8]],
 [[1,2],[3,4]]]


image2 = [[[2,1],[4,3]],
 [[2,1],[4,3]],
 [[2,1],[4,3]]]



# mppmi(image1, image2)

im3 = np.random.rand(3, 10, 10)
print(im3)
imageMPMF(im3)


