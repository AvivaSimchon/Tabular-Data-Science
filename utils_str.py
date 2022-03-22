"""
Utils file for string popular operations.

Basic assumptions:
1. file name is consist of a string NAME, a '.' and a string EXTENSION.
"""

from pandas import to_datetime


# ----------------------------------------------------------------------------------------------------------------------
# full_path
# Given a directory and a filename, create the full path.
# ----------------------------------------------------------------------------------------------------------------------
def full_path(dir_in, dir_or_file_name):
    return dir_in + '\\' + dir_or_file_name


# ----------------------------------------------------------------------------------------------------------------------
# remove_file_ext
# Given a file name with its extension, extract and return the file name without the extension.
# ----------------------------------------------------------------------------------------------------------------------
def remove_file_ext(file_name):
    return file_name[:file_name.find('.')]


# ----------------------------------------------------------------------------------------------------------------------
# get_file_ext
# Given a file name, return its file extension.
# ----------------------------------------------------------------------------------------------------------------------
def get_file_ext(file_name):
    return file_name[file_name.find('.')+1:]


# ----------------------------------------------------------------------------------------------------------------------
# extend_file_name
# Extend the given file name with an additional phrase.
# ----------------------------------------------------------------------------------------------------------------------
def extend_file_name(file_name, phrase):
    file_name_no_ext = remove_file_ext(file_name)
    file_ext = get_file_ext(file_name)
    return file_name_no_ext + phrase + '.' + file_ext


# ----------------------------------------------------------------------------------------------------------------------
# replace_file_ext
# Replace the file extension
# ----------------------------------------------------------------------------------------------------------------------
def replace_file_ext(file_name, new_ext):
    return file_name[:file_name.find('.')] + '.' + new_ext


# ----------------------------------------------------------------------------------------------------------------------
# print_main_process_title
# Print the title of a given main process.
# ----------------------------------------------------------------------------------------------------------------------
def print_main_process_title(title):
    print('===================================================================')
    print(title)
    print('===================================================================')


# ----------------------------------------------------------------------------------------------------------------------
# divide_list_to_sub_lists_by_prefix
# Given a list of strings, divide the list to sub-lists by the given prefix
# ----------------------------------------------------------------------------------------------------------------------
def divide_list_to_sub_lists_by_prefix(in_list, prefix_end):
    sub_lists = list()
    while in_list:
        elem = in_list.pop(0)
        sub_list = [elem]

        # Compute the prefix of the current element and collect all the remaining elements having this prefix.
        prefix = elem[:elem.find(prefix_end)]
        sub_list = sub_list + [e for e in in_list if prefix in e]
        sub_lists.append(sub_list)
        in_list = [e for e in in_list if prefix not in e]

    return sub_lists


# ----------------------------------------------------------------------------------------------------------------------
# find_index_of_last_element_with_prefix
# Iterate over the given list and find the index of the last element that contains the given prefix.
# ----------------------------------------------------------------------------------------------------------------------
def find_last_element_with_prefix(lst, prefix):
    index = -1
    num = len(lst)
    for i in range(num):
        if prefix in lst[i]:
            index = i
    return index


# ----------------------------------------------------------------------------------------------------------------------
# get_valid_timestamp
# In case the given timestamp format is implicitly convert to datetime, we have to covert it into a string
# representation which will not be implicitly converted since MLflow does not support datetime\Timestamp
# objects. The final date representation is of the format '%Y%m%d'.
# ----------------------------------------------------------------------------------------------------------------------
def get_valid_timestamp(ts_str, ts_format='%d/%m/%Y'):
    ts_datetime = to_datetime(ts_str, format=ts_format)
    return ts_datetime.strftime('%Y%m%d')