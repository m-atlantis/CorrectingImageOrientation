import sys

from cluster_image_feature_vectors import cluster
from correction import predict, train, extract
from feature_detection import generate_features
from util import rotate_images

# The following project is created by following the guide on https://www.kaggle.com/code/itsshuvra/correcting-image-rotation/notebook
# using the dataset AID from https://captain-whu.github.io/AID/
rotated_path = "Rotated_Images/"

if __name__ == "__main__":
    for i, arg in enumerate(sys.argv):
        if i > 0:
            if arg == "-rotate" or arg == "-r":
                if len(sys.argv) > i + 1:
                    print("[INFO] Rotating images in '" + sys.argv[i + 1] + "'")
                    rotate_images(rotated_path + sys.argv[i + 1])
                    break
                else:
                    print("[ERROR] -rotate | -r : Needs the subfolder to rotate images from as the second argument.")
            if arg == "-extract" or arg == "-e":
                if len(sys.argv) > i + 1:
                    print("[INFO] Extracting features from images in '" + sys.argv[i + 1] + "'")
                    extract(rotated_path + str(sys.argv[i + 1]))
                    break
                else:
                    print("[ERROR] -extract | -e : Needs the sub-folder with images as the second argument.")
            if arg == "-train" or arg == "-t":
                print("[INFO] Initialize training")
                train()
                break
            if arg == "-predict" or arg == "-p":
                if len(sys.argv) > i + 1:
                    if len(sys.argv) > i + 2:
                        print("[INFO] Predicting on " + sys.argv[i + 2] + " random image(s) from '" + sys.argv[i + 1] + "'")
                        predict(sys.argv[i + 1], int(sys.argv[i + 2]))
                    else:
                        print("[INFO] Predicting on 10 random images from '" + sys.argv[i + 1] + "'")
                        predict(sys.argv[i + 1])
                    break
                print("[INFO] Predicting on 10 random images")
                predict(rotated_path)
                break
            if arg == "-features" or arg == "-f":
                # Remember to run the rotate.py file first.
                if len(sys.argv) > i + 1:
                    print("[INFO] Extracting features from images in '" + sys.argv[i + 1] + "'")
                    generate_features(str(sys.argv[i + 1]))
                    break
                else:
                    print("[ERROR] -features | -f : Needs the path to a folder with .jpg images as first argument.")
                break
            if arg == "-cluster" or arg == "-c":
                # Remember to run the rotate.py file first, and then find features using the -f command.
                if len(sys.argv) > i + 1:
                    print("[INFO] Clustering features '" + sys.argv[i + 1] + "'")
                    cluster(sys.argv[i + 1])
                    break
                else:
                    print("[ERROR] -cluster | -c : Needs the name of an image, without it's extension as the argument.")

                break
            else:
                print("Use one of the following commands:")
                print("\t-rotate * | -r * : Rotates images in 'Images/*' and places the result in the 'Rotated_Images/*' folder.")
                print("\t-extract * | -e * : Extracts features from images in 'Rotated_Images/*' folder.")
                print("\t-predict * * | -p * * | -predict * | -p * | -predict | -p:  Predicts on either 10 random images from the "
                      "'Rotated_Images/*' folder, or a given amount of files from a given folder.")
                print("")
                print("\tThese two commands requires that the 'rotate.py' file has been run")
                print("\t-features * | -f * : Extracts features from .jpg images in a given folder.")
                print("\t-cluster * | -c * : Cluster features using the generated feature vectors from the -f command.")
