import sys
import os
import numpy as np
import pytest
from tests import _PATH_SRC_DATA, _PATH_DATA

sys.path.append(_PATH_SRC_DATA)
from make_dataset import load_mnist

# extract data directory and names of the train and test data files
data_dir = os.path.join(_PATH_DATA, "raw")
data_fpaths = []
for i in range(8):
    data_file = "train_" + str(i) + ".npz"
    data_fpaths.append(os.path.join(data_dir, data_file))
data_fpaths.append(os.path.join(data_dir, "test.npz"))

# check for missing data files
missing_data_flag = False
for fpath in data_fpaths:
    if not os.path.exists(fpath):
        missing_data_flag = True
        break

# skip the test if the data is not present
@pytest.mark.skipif(missing_data_flag, reason="Data files not found")
def test_data():
    data = load_mnist(data_dir)
    train_images, train_labels, test_images, test_labels = data

    # test that we load the right amount of train and test images
    assert len(train_images) == 40000, "Dataset did not have the correct number of images"
    assert len(test_labels) == 5000, "Dataset did not have the correct number of labels"

    # test that all the images have the correct shape
    for set in [train_images, test_images]:
        for img in set:
            assert img.shape == (1, 28, 28) or img.shape == (
                28,
                28,
            ), "Not every image has shape (1, 28, 28)"

    # test that all the images correspond to a label
    assert len(train_labels) == len(
        train_images
    ), "Train labels and images are not the same number"
    assert len(test_labels) == len(
        test_images
    ), "Test labels and images are not the same number"
