#!/usr/bin/python
import os
import numpy as np
import codecs

def get_webkb_files():
    """
    Retrieve all WebKB data and return two arrays - the first stores the files,
    as strings, the latter holds their corresponding class labels
    :return: nparray of strings
    """
    current_dir = os.getcwd()
    student_files = get_files(current_dir + "/data/webkb/student")
    faculty_files = get_files(current_dir + "/data/webkb/faculty")
    course_files = get_files(current_dir + "/data/webkb/course")
    project_files = get_files(current_dir + "/data/webkb/project")

    files = student_files + faculty_files + course_files + project_files
    file_classes = [1] * len(student_files) + [2] * len(faculty_files) +\
                   [3] * len(course_files) + [4] * len(project_files)
    return np.array(files), np.array(file_classes)


def get_files(top_dir):
    """
    Traverse the directory structure and retrieve all files in the top_dir
    directory and all other directories inside it
    :param top_dir:
    :return: list of strings
    """
    files = []
    for root, dirs, fs in os.walk(top_dir, topdown=True):
        for name in fs:
            files.append(codecs.open(root + "/" + name, "r",
                            "us-ascii", "ignore").read().decode("us-ascii"))
    return files

