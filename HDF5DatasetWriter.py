import os

import h5py


class HDF5DatasetWriter:
    def __init__(self, dims, output_path, data_key="images", buffer_size=1000):
        # Ensure the hdf5 file doesn't already exist
        if os.path.exists(output_path):
            raise ValueError("The supplied ‘output_path‘ already exists and cannot be overwritten. Manually delete "
                             "the file before continuing.", output_path)

        # open the HDF5 database for writing and create two datasets:
        # The first database is for to store raw images(extracted features) and
        # another one is for label of the images in integer format.

        self.db = h5py.File(output_path, "w")
        self.data = self.db.create_dataset(data_key, dims,
                                           dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0],),
                                             dtype="int")

        # store the buffer size, then initialize the buffer itself along with the index into the datasets
        self.buffer_size = buffer_size
        self.buffer = {"data": [], "labels": []}
        self.idx = 0

    def add(self, rows, labels):
        """Add the rows and labels until the buffer size"""
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)
        # after reached to the buffer
        # size we have flush to reset it.
        if len(self.buffer["data"]) >= self.buffer_size:
            self.flush()

    def flush(self):
        """used to flush the data"""
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def store_class_labels(self, class_labels):
        """Creates a dataset and store the class label names in it"""
        dt = h5py.special_dtype(vlen=str)
        label_set = self.db.create_dataset("label_names", (len(class_labels),), dtype=dt)
        label_set[:] = class_labels

    def close(self):
        """Flushes data and closes the database"""
        if len(self.buffer["data"]) > 0:
            self.flush()

        self.db.close()
