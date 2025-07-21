# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Input
# from tensorflow.keras.layers import Lambda
import numpy as np

from net.siamese_net import build_siamese_model
from net import config
from net import utils



# import dataset
# print("[INFO] loading MNIST")
# utils.import_data(config.DATA_PATH)
# (trainX, trainY), (testX, testY) = utils.import_data()
# trainX = trainX / 255.0
# testX = testX / 255.0

# trainX = np.expand_dims(trainX, axis=-1)
# testX = np.expand_dims(testX, axis=-1)

