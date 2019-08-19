import os
import h5py
import numpy as np


# TODO - encode to binary
def setup_hdf5(file_name, n_encoded, n_labels=1, encoded_dtype='float32'):
    '''
    create an HDF5 file with name `file_name` for writing and setup groups and
    datasets for 'images' and 'labels' with `n_encoded` image shape and
    `n_labels` label shape (almost certainly 1).
    '''
    if os.path.exists(file_name):
        os.remove(file_name)

    f = h5py.File(file_name, 'w')
    grp = f.create_group('encoded')
    grp.create_dataset(
        'images', (0, n_encoded), dtype=encoded_dtype, compression='gzip',
        maxshape=(None, n_encoded)
    )
    grp.create_dataset(
        'labels', (0, n_labels), dtype='uint8', compression='gzip',
        maxshape=(None, n_labels)
    )
    return f


def add_batch_to_hdf5(f, encoded_set, labels_set):
    '''
    write a batch of data to an HDF5 file with h5py handle `f` and numpy arrays
    `encoded_set` for the compressed images and `labels_set` for the
    corresponding labels.
    '''
    assert len(encoded_set) == len(labels_set), "data length mismatch"
    existing_examples = np.shape(f['encoded/images'])[0]
    total_examples = len(encoded_set) + existing_examples
    f['encoded/images'].resize(total_examples, axis=0)
    f['encoded/labels'].resize(total_examples, axis=0)
    f['encoded/images'][existing_examples: total_examples] = encoded_set
    f['encoded/labels'][existing_examples: total_examples] = labels_set
    return total_examples
