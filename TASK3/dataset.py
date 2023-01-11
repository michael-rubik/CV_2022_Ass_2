# TUWIEN - WS2022 CV: Task3 - Scene recognition using Bag of Visual Words
# *********+++++++++*******++++INSERT GROUP NO. HERE
from typing import List, Tuple
import numpy as np
import cv2
import os


class SceneDataset():
    images = []         # list of images
    labels = []         # list of labels of images
    class_names = ""    # list with of class names (folder names) 

    def __init__(self, path: str):
        """
        Loop through all subfolders within the given 'path', get all images per folder,
        save the images in gray scale and normalize the image values between 0 and 1.
        The label of an image is the current subfolder (eg. value between 0-9 when using 10 classes)
        HINT: os.walk(..), cv2.imread(..)
        path : path to dataset - string
        """

        # student_code start
        # raise NotImplementedError("TO DO in dataset.py")

        img_data = []
        labels = []
        dirs = []

        # Get list of (dirpath, dirnames, filenames)-tuples 
        # for every subfolder under path/ including folder path/ itself
        dirtree = os.walk(path, topdown=True, onerror=None, followlinks=False)
        
        # Class labels should start at 0
        class_label_idx = 0

        # For every entry in the directory tree (distree_entries)
        for entry in dirtree:
            # If dirpath is path itself, the dirnames are the names of path/ subfolders (= class names)
            # save name to list
            if entry[0] == path:
                dirs.append(entry[1])

            # If dirpath is not path we are looking at a subfolder of path/
            # thus filenames contains a list of the names of images in the subfolder 
            else:
                image_dir_path = entry[0]  
                image_names = entry[2]

                # For each image in subfolder build the valid path
                # read, grayscale, normalize and save image to list
                # save the index of the associated class label to list
                for image_name in image_names:
                    image_path = os.path.join(image_dir_path, image_name)
                    image = cv2.imread(image_path)
                    grayscale_normal_image = cv2.normalize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None, 0, 255).astype("uint8")
                    img_data.append(grayscale_normal_image)
                    labels.append(class_label_idx)

            class_label_idx += 1
        # student_code end

        """
        save as local parameters:
        images : list of images [num_of_images x n x m] - float
        labels : list of belonging label per image [num_of_images x 1] - int
        class_names : list of names of the subfolders (classes) [num_of_classes x 1] - string
        """
        self.images = img_data
        self.labels = labels
        self.class_names = dirs

    def get_data(self) -> Tuple[List[np.ndarray], List[int]]:
        return self.images, self.labels

    def get_class_names(self) -> List[str]:
        return self.class_names
