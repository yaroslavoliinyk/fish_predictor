import matplotlib
import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from datasets.datasetloader import DatasetLoader
from preprocessing.preprocessor import SimplePreprocessor
# from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.datasets import cifar10
from imutils import paths


matplotlib.use("Agg")


class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        # first CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # second CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        # return the constructed network architecture
        return model


# A block for preparing image to utilize Neural Network on images
def convert(source, dest):
    for imagePath in sorted(list(paths.list_images(source))):
        # load image,  preprocess, save
        image = cv2.imread(imagePath)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # dsize = (28, 28)
        dsize = (32, 32)
        image = cv2.resize(image, dsize, interpolation=cv2.INTER_AREA)
        if not os.path.exists(dest):
            os.mkdir(dest)
        fish_class = imagePath.split('/')[-2]
        fish_id = imagePath.split('/')[-1]
        if not os.path.exists(dest + "/{}".format(fish_class)):
            os.mkdir(dest + "/{}".format(fish_class))
        cv2.imwrite(dest + "/{0}/{1}".format(fish_class, fish_id), image)


# convert("archive/NA_Fish_Dataset", "archive_converted_32"
img_folder = "data/archive/NA_Fish_Dataset"
preprocesssor = SimplePreprocessor(32, 32)
dsl = DatasetLoader(preprocesssor)
imagePaths = sorted(list(paths.list_images("{}".format(img_folder))))

# Reading images from disk and storing in RAM parameter
def read(convFolder="archive_converted_32"):
    images = []
    labels = []
    for imagePath in sorted(list(paths.list_images("{}".format(convFolder)))):
        image = cv2.imread(imagePath)
        image = img_to_array(image)
        images.append(image)
        labels.append(imagePath.split("/")[-2])
    return images, labels


def fit_model():
    images, labels = read()
    images = np.array(images, dtype="float") / 255.0
    labels = np.array(labels)
    lb = LabelBinarizer().fit(labels)
    labels = lb.fit_transform(labels)
    # split the data
    trainX, testX, trainY, testY = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=42
    )
    print("[INFO] Training Network")
    model = MiniVGGNet.build(width=32, height=32, depth=3, classes=9)
    plot_model(model, to_file="lenet.png", show_shapes=True)
    model.compile(loss="binary_crossentropy", optimizer="adam")
    H = model.fit(
        trainX,
        trainY,
        validation_data=(testX, testY),
        batch_size=8,
        epochs=30,
        verbose=1,
    )
    print("[INFO] Evaluating Network")
    preds = model.predict(testX, batch_size=64)
    print(
        classification_report(
            testY.argmax(axis=1), preds.argmax(axis=1), target_names=lb.classes_
        )
    )
    # save the model to disk
    model.save("fish.hdf5")


fit_model()
