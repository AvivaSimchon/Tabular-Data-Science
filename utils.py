"""
Utils file for popular operations.
"""

import os
import shutil

from numpy import array


# ----------------------------------------------------------------------------------------------------------------------
# clear_dir
# Given a directory clear all its files and sub-directories
# ----------------------------------------------------------------------------------------------------------------------
def clear_dir(directory):
    if not os.path.exists(directory):
        return
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


# ----------------------------------------------------------------------------------------------------------------------
# split_sequence_into_t_x_y_subsequences
# Get a multivariate sequence that contains timestamps, features and targets and split it into timestamps, features
# and targets subsequences of given length.
# ----------------------------------------------------------------------------------------------------------------------
def split_sequence_into_t_x_y_subsequences(sequence, subsequence_len):

    # Clarifications:
    # t - the timestamp at the last day of the subsequence
    # X - the feature vector of the subsequence
    # y - the label of the last day of the subsequence
    t, X, y = list(), list(), list()
    for curr_index in range(len(sequence)):

        # Find the end of the current subsequence (pattern) and break if it exceeds the sequence (dataset).
        end_index = curr_index + subsequence_len
        if end_index > len(sequence):
            break

        # Gather the input and output parts of the subsequence (pattern).
        seq_t = sequence[end_index - 1, 0]
        seq_x = sequence[curr_index:end_index, 1:-1]
        seq_y = sequence[end_index - 1, -1]
        t.append(seq_t)
        X.append(seq_x)
        y.append(seq_y)

    return array(t), array(X), array(y)


# ----------------------------------------------------------------------------------------------------------------------
# list_splitter
# Split a given list by a given ratio
# ----------------------------------------------------------------------------------------------------------------------
def list_splitter(list_to_split, ratio):
    elements = len(list_to_split)
    middle = int(elements * ratio)
    return list_to_split[:middle], list_to_split[middle:]


# ----------------------------------------------------------------------------------------------------------------------
# remove_duplications
# Given a list, remove all the duplicated elements
# ----------------------------------------------------------------------------------------------------------------------
def remove_duplications(in_list):
    return list(dict.fromkeys(in_list))