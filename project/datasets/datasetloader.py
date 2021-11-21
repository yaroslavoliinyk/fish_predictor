import numpy as np
import cv2
import os
import glob


class DatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        data = []
        labels = []

        for (i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)
            # treat our processed image as a "feature vector"
            # by updating the data list followed by the labels
            data.append(image)
            labels.append(label)
            # show an update every ‘verbose‘ images
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))
        # return a tuple of the data and labels
        return (np.array(data), np.array(labels))

    def load_from_disk(self, trainPath="data/train/", testPath="data/test/",):
        pass

    def save(
        self,
        trainX,
        testX,
        trainY,
        testY,
        trainPath="data/train/",
        testPath="data/test/",
    ):
        # remove previous saved train and test images
        train_file_list = glob.glob(os.path.join(trainPath, "*"))
        for f in train_file_list:
            os.remove(f)
        test_file_list = glob.glob(os.path.join(testPath, "*"))
        for f in test_file_list:
            os.remove(f)
        # overwrite train images where image name == trainY label
        for (i, train_img) in enumerate(trainX):
            full_path = trainPath + trainY[i] + "_" + str(i) + ".jpg"
            cv2.imwrite(full_path, train_img)
        # overwrite test images where image name == testY label
        for (i, test_img) in enumerate(testX):
            full_path = testPath + testY[i] + "_" + str(i) + ".jpg"
            cv2.imwrite(full_path, test_img)
