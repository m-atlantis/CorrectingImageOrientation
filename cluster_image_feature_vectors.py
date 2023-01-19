#################################################
# This script reads image feature vectors from a folder
# and saves the image similarity scores in json file
# by Erdem Isbilen - December/2019
#################################################

#################################################
# Imports and function definitions
#################################################

# Glob for reading file names in a folder
import glob
# json for storing data in json file
import json
import os.path
# Time for measuring the process time
import time

# Numpy for loading image feature vectors from file
import numpy as np
# Annoy and Scipy for similarity calculation
from annoy import AnnoyIndex
from scipy import spatial


#################################################

#################################################
# This function; 
# Reads all image feature vectors stored in /feature-vectors/*.npz
# Adds them all in Annoy Index
# Builds ANNOY index
# Calculates the nearest neighbors and image similarity metrics
# Stores image similarity scores with name in a json file
#################################################
def cluster(master_img_name):
    start_time = time.time()

    print("---------------------------------")
    print("Step.1 - ANNOY index generation - Started at %s" % time.ctime())

    # Defining data structures as empty dict
    file_index_to_file_name = {}
    file_index_to_file_vector = {}

    # Configuring annoy parameters
    dims = 1792
    n_nearest_neighbors = 360
    trees = 10000

    # Reads all file names which stores feature vectors
    allfiles = glob.glob('feature-vectors/*.npz')

    master_file_name = None
    master_vector = None
    master_index = None
    t = AnnoyIndex(dims, metric='angular')

    for file_index, i in enumerate(allfiles):
        # Reads feature vectors and assigns them into the file_vector
        file_vector = np.loadtxt(i)

        # Assigns file_name, feature_vectors and corresponding product_id
        file_name = os.path.basename(i).split('.')[0]
        file_index_to_file_name[file_index] = file_name
        file_index_to_file_vector[file_index] = file_vector

        # Adds image feature vectors into annoy index
        t.add_item(file_index, file_vector)

        if file_name == master_img_name or file_name == master_img_name.replace(".jpg", ""):
            # Assigns master file_name, image feature vectors and product id values
            master_file_name = file_index_to_file_name[file_index]
            master_vector = file_index_to_file_vector[file_index]
            master_index = file_index

    # Builds annoy index
    t.build(trees)

    print("Step.1 - ANNOY index generation - Finished")
    print("Step.2 - Similarity score calculation - Started ")

    named_nearest_neighbors = []

    # Loops through all indexed items
    # for i in file_index_to_file_name.keys():

    # Calculates the nearest neighbors of the master item
    nearest_neighbors = t.get_nns_by_item(master_index, n_nearest_neighbors)

    # Loops through the nearest neighbors of the master item
    for j in nearest_neighbors:
        # Assigns file_name, image feature vectors and product id values of the similar item
        neighbor_file_name = file_index_to_file_name[j]
        neighbor_file_vector = file_index_to_file_vector[j]

        # Calculates the similarity score of the similar item
        similarity = 1 - spatial.distance.cosine(master_vector, neighbor_file_vector)
        rounded_similarity = int((similarity * 10000)) / 10000.0

        # Appends master product id with the similarity score
        # and the product id of the similar items
        named_nearest_neighbors.append({
            'similarity': rounded_similarity,
            'master_name': master_file_name,
            'closest_name': neighbor_file_name
        })

    print("Step.2 - Similarity score calculation - Finished ")

    # Writes the 'named_nearest_neighbors' to a json file
    with open('json-files/nearest_neighbors.json', 'w') as out:
        json.dump(named_nearest_neighbors, out)

    print("Step.3 - Data stored in 'nearest_neighbors.json' file ")
    print("--- Process completed in %.2f minutes ---------" % ((time.time() - start_time) / 60))
