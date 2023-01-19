import os
import random

import cv2
import imutils
import numpy as np
import progressbar
from imutils import paths


def rotate_images(sub_path):
    # Initialize the dataset location and storing location
    dataset_path = "Images/" + sub_path
    new_rotated_path = sub_path

    # Load (10000) images from dataset path into a list and shuffle them
    image_paths = list(paths.list_images(dataset_path))[:10000]
    random.shuffle(image_paths)

    # Init a dict to keep track of the rotations
    angles = {}

    # Init a progress bar, to easier follow the progress
    widgets = ["[INFO] Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    # Define the max val in progress bar with the total images from our dataset so that it'll end at that iteration.
    progress_bar = progressbar.ProgressBar(maxval=len(image_paths), widgets=widgets).start()

    # Create folder if not exists
    if not os.path.exists(new_rotated_path):
        print("[INFO] Creating destination sub-folder: " + new_rotated_path)
        os.makedirs(new_rotated_path)

    # Perform rotations
    for (idx, path) in enumerate(image_paths):
        # Randomly choose a rotation angle, and load the image
        angle = np.random.choice([0, 90, 180, 270])
        image = cv2.imread(path)

        # if the image is None
        if image is None:
            continue

        # rotate the image based on the selected angle, then construct
        # the path to the base output directory
        image = imutils.rotate_bound(image, angle)
        base = os.path.sep.join([new_rotated_path, str(angle)])

        # if the base path does not exist already, create it
        if not os.path.exists(base):
            os.makedirs(base)

        # extract the image file extension, then construct the full path
        # to the output file
        ext = path[path.rfind("."):]
        output_path = [base, "image_{}{}".format(str(angles.get(angle, 0)).zfill(5), ext)]
        output_path = os.path.sep.join(output_path)

        # save the image
        cv2.imwrite(output_path, image)
        # update the count for the angle
        c = angles.get(angle, 0)
        angles[angle] = c + 1
        progress_bar.update(idx)

    # Finish progress bar
    progress_bar.finish()

    # loop over the angles and display counts for each of them
    for angle in sorted(angles.keys()):
        print("[INFO] angle={}: {:,}".format(angle, angles[angle]))
    # loop over the angles and display counts for each of them
    for angle in sorted(angles.keys()):
        print("[INFO] angle={}: {:,}".format(angle, angles[angle]))


def get_dir_size(path):
    """Counts the number of files in a given directory."""
    return len([e for e in os.listdir(path) if os.path.isfile(os.path.join(path, e))])
