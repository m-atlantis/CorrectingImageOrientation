import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Used to stop tensorflow from bloating the terminal, find a beter way.

import pickle
import random

import cv2
import h5py
import imutils
import numpy as np
import progressbar
from imutils import paths
from keras.applications import imagenet_utils
from keras.utils import load_img, img_to_array
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from HDF5DatasetWriter import HDF5DatasetWriter
from VGGNet16 import VGGNet16
from util import get_dir_size

hdf5_path = "correcting_rotation_dataset.hdf5"
# to serialize the model location initialize
model_path = "Model/orientation_correction_classifier.cpickle"

# initialize vgg16 trained model weights path location
weight_path = "Weight/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

# initialize batch size and buffer size for predicting in batch and to store the buffer sizes images into file.
batch_size = 32
buffer_size = 1000


def extract(sub_path):
    """Runs the VGGNet16 model on the weights, and extract the features from the model."""
    # Here, we load all images path from given rotated_path location.
    print("[INFO] Loading images...")
    image_paths = list(paths.list_images(sub_path))
    # After loading all images, shuffle them to achieve randomness
    random.shuffle(image_paths)

    # For getting the label or class for each class we extract each label from each image path. Because, we have class label folder.
    labels = [p.split(os.path.sep)[-2] for p in image_paths]
    # Here, we encode the class level into 0,1,2,3 numeric value.
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    dataset = HDF5DatasetWriter((len(image_paths), 512 * 7 * 7), hdf5_path, data_key="features", buffer_size=buffer_size)

    # here, we store all the class names that we are going to predict. [0,90,180,270]
    dataset.store_class_labels(le.classes_)

    # Init the progress bar
    widgets = ["[Info] Extracting Features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(image_paths), widgets=widgets).start()

    # Ensure weights exist
    if not os.path.exists(weight_path):
        raise ValueError("[ERROR] No weights found at '" + weight_path + "'. Extraction terminated.")
    model = VGGNet16.build(weight_path)
    model.summary()

    # Here, we are taking the images from image_paths by batch size. And extract all batch images feature from VGG-16 model
    for idx in np.arange(0, len(image_paths), batch_size):

        # we define batch_paths for store all images path from image_paths by the batch size that we define before.
        batch_paths = image_paths[idx:idx + batch_size]
        # It is same here above line, but we actually store the labels of each batch size images.
        batch_labels = labels[idx:idx + batch_size]
        # Initializes the batch_images for storing the all batch size images.
        batch_images = []

        # loop over the images and labels in the current batch
        for (j, imagePath) in enumerate(batch_paths):
            # we resized all images to 224x224 pixels
            image = load_img(imagePath, target_size=(224, 224))
            image = img_to_array(image)

            # Adds extra dimension for each image: (224x224x3) => (1x224x224x3)
            image = np.expand_dims(image, axis=0)
            image = imagenet_utils.preprocess_input(image)

            # Add the image to the batch
            batch_images.append(image)

        # Convert list to numpy array with vertical stack.
        batch_images = np.vstack(batch_images)
        # Extract feature from VGG-16 models last layer (flatten layer) by batch.
        features = model.predict(batch_images, batch_size=batch_size)

        # Reshapes the features so that each image is represented by a flattened feature vector of the ‘MaxPooling2D‘ outputs
        features = features.reshape((features.shape[0], 512 * 7 * 7))

        # Add the features and labels to HDF5 dataset
        dataset.add(features, batch_labels)
        pbar.update(idx)

    # Close the dataset
    dataset.close()
    pbar.finish()


def train():
    # Init the hdf5 dataset path where we store the final images into hdf5 file
    db = hdf5_path

    # open the HDF5 database for reading
    db = h5py.File(db, "r")

    # Init the index of the 75% of the dataset into "idx"
    idx = int(db["labels"].shape[0] * 0.75)

    print("[INFO] Tuning hyperparameters")
    params = {"C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}

    # Validation
    model = GridSearchCV(LogisticRegression(max_iter=300), params, cv=3, n_jobs=1)
    # Train the model using 75% of the data among total data
    model.fit(db["features"][:idx], db["labels"][:idx])

    # Print the best parameters values that we got after training the model
    print("[INFO] Best hyperparameters: {}".format(model.best_params_))

    print("[INFO] Evaluating")
    # Predicting the class
    predictions = model.predict(db["features"][idx:])

    # Print the classification report after getting the prediction from the model by test data.
    print("[INFO] Result:")
    # report = classification_report(db["labels"][idx:], predictions, target_names=db["label_names"])
    report = classification_report(db["labels"][idx:], predictions)
    print(report)

    # Serialize the model to disk so that we can use this model next time for prediction
    print("[INFO] Saving model")
    f = open(model_path, "wb")
    f.write(pickle.dumps(model.best_estimator_))
    f.close()

    # close the database
    db.close()


def predict(dataset=None, size=10):
    # Init paths
    db = hdf5_path
    model = model_path

    if dataset is None:
        raise ValueError("No dataset selected.")

    # Load the label names from the HDF5 dataset
    db = h5py.File(db)
    # Convert the label_names into integers
    label_names = [int(angle) for angle in db["label_names"][:]]
    # Close the database connection
    db.close()

    # Grab the paths to the testing images and randomly sample them
    print("[INFO] Sampling images...")
    # Load all the images from 'Rotated_Images' path
    image_paths = list(paths.list_images(dataset))
    # Choose 10 images randomly for testing without replacement
    image_paths = np.random.choice(image_paths, size=(min(size, get_dir_size(dataset)),), replace=False)

    # Load the VGG16 network for extracting features for each testing images
    print("[INFO] Loading network...")
    vgg = VGGNet16.build(weight_path)

    # Load the orientation model for prediction
    print("[INFO] Loading model...")
    model = pickle.loads(open(model, "rb").read())

    # Loop over the chosen 10 images to predict
    for imagePath in image_paths:
        orig = cv2.imread(imagePath)
        # Read the image and resize it to 224x224x3
        image = load_img(imagePath, target_size=(224, 224))

        # Add extra dimension: (224x224x3) => (1x224x224x3)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        # Pass the image through the VGG16 network to get the feature vector and reshape the features into a 1D array for prediction
        features = vgg.predict(image)
        features = features.reshape((features.shape[0], 512 * 7 * 7))

        # Pass the CNN features through the classifier to obtain the orientation predictions
        angle = model.predict(features)
        angle = label_names[angle[0]]

        # Correct the image using the orientation prediction
        rotated = imutils.rotate_bound(orig, 360 - angle)
        orig = cv2.resize(orig, (300, 300))
        rotated = cv2.resize(rotated, (300, 300))

        # Display the original and corrected images for a side-by-side comparison
        cv2.imshow("Original", orig)
        cv2.imshow("Corrected", rotated)
        cv2.waitKey(0)  # Go to next image by pressing any key
    cv2.destroyAllWindows()
