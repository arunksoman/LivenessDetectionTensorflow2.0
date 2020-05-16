"""
Uage:
if you did not created the folder structure yet. You have to run this file once.
Also it contains absolute path to models we can use them enormously everywhere importing configurations.
"""

import os
# import glob

"""
BASE_DIR will return absolute path to your Project directory
os.path.join joins the strings correctly as a path structure.

"""
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "Videos")
if not os.path.exists(VIDEO_DIR):
    os.mkdir(VIDEO_DIR)

DATASET_DIR = os.path.join(BASE_DIR, "Dataset_Detection")
if not os.path.exists(DATASET_DIR):
    os.mkdir(DATASET_DIR)

FAKE_DIR = os.path.join(DATASET_DIR, "Fake")
if not os.path.exists(FAKE_DIR):
    os.mkdir(FAKE_DIR)

FAKE_DIR = os.path.join(DATASET_DIR, "Fake")
if not os.path.exists(FAKE_DIR):
    os.mkdir(FAKE_DIR)

REAL_DIR = os.path.join(DATASET_DIR, "Real")
if not os.path.exists(REAL_DIR):
    os.mkdir(REAL_DIR)

"""
Path to SSD Model(Model contains weighta and bias) and prototxt file(contains architecture of network).
"""
SSDModel_DIR = os.path.join(BASE_DIR, "SSDModels")
if not os.path.exists(SSDModel_DIR):
    os.mkdir(SSDModel_DIR)

CAFFEMODEL_PATH =  os.path.join(SSDModel_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
PROTXT_PATH = os.path.join(SSDModel_DIR, "deploy.prototxt")
