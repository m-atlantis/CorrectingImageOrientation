import argparse
import os

import cv2
import imutils
import numpy as np
import progressbar

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to an image file")
args = vars(ap.parse_args())

# load the image from disk
path = args["image"]
image = cv2.imread(path)

# Init a progress bar, to easier follow the progress
widgets = ["[INFO] Rotating imagesI: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
# Define the max val in progress bar with the total images from our dataset so that it'll end at that iteration.
progress_bar = progressbar.ProgressBar(maxval=(360 / 10), widgets=widgets).start()

# loop over the rotation angles
index = 0
for angle in np.arange(0, 360, 10):
    rotated = imutils.rotate_bound(image, angle)

    # Used to show the images as they are generated
    # cv2.imshow("Rotated (Problematic)", rotated)
    # cv2.waitKey(0)

    # extract the image file extension, then construct the full path
    # to the output file
    ext = path[path.rfind("."):]
    output_path = [os.path.dirname(path), "image_{}{}".format(str(index).zfill(5), ext)]
    output_path = os.path.sep.join(output_path)

    # save the image
    cv2.imwrite(output_path, rotated)
    # update the progress bar
    progress_bar.update(index)
    index += 1

# Finish progress bar
progress_bar.finish()
