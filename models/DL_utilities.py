# main libs
import pandas as pd
import numpy as np
# data preparation libs
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from keras.utils import to_categorical
# model libs
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import BatchNormalization, LeakyReLU, Input,ReLU
from keras.optimizers import Adamax,SGD
from keras.regularizers import l1,l2
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import regularizers
import keras
import tensorflow as tf
#import keras.backend.tensorflow_backend as K
from keras.models import Sequential, Model
# visualisation libs
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
plt.ioff()


class Generator_amputor_team(keras.utils.Sequence):
    """
    data generator that randomly removes on champ from team and 
    create an output vector that is this champion
    """
    def __init__(self, X, N_champs, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.X = X.copy()
        self.shuffle = shuffle
        self.on_epoch_end()
        self.N_champs = N_champs

    def __len__(self):
        """
        number of batchs par epochs
        """
        return int(np.floor(len(self.X) / self.batch_size))


    def __getitem__(self, index_batch):
        """
        generates one batch of data
        batch goes from batch_id * batchsize -> (batch_id + 1)* batchsize OR end of X
        """
        indexes_batch = self.X.index[
            index_batch*self.batch_size:max((index_batch+1)*self.batch_size, len(self.X))
        ]
        # Generate data
        X, Y = self.__data_generation(indexes_batch)

        return X, Y

    def on_epoch_end(self):
        'Shuffles indexes after each epoch'
        if self.shuffle == True:
            self.X = self.X.sample(frac=1).reset_index(drop=True)

    def __data_generation(self, indexes_batch):
        """
        generates X, y from X
        """

        # remove randomly one column
        col_removed = np.random.choice(self.X.columns)
        kept_col = [c for c in self.X.columns if c != col_removed]

        X = self.X.loc[indexes_batch, kept_col].values
        Y = self.X.loc[indexes_batch, col_removed].values


        # dense to "one hot"
        X = tf.keras.utils.to_categorical(X, num_classes=self.N_champs).sum(axis=1)
        Y = tf.keras.utils.to_categorical(Y, num_classes=self.N_champs)
        return X, Y 

class Generator_amputor_game(Generator_amputor_team):
  """
  essentially a Generator_amputor_team except that
  instead of taking team by team, it take two teams.
  encoding is 1 for team of the amputee and -1 for ennemy team
  """
  def __data_generation(self, indexes_batch):
        """
        generates X, y from X
        """

        # remove randomly one column
        col_removed = np.random.choice(self.X.columns)
        kept_col = [c for c in self.X.columns if c != col_removed]

        X = self.X.loc[indexes_batch, kept_col].values
        Y = self.X.loc[indexes_batch, col_removed].values


        # dense to "one hot"
        X = tf.keras.utils.to_categorical(X, num_classes=N_champs).sum(axis=1)
        Y = tf.keras.utils.to_categorical(Y, num_classes=N_champs)
        return X, Y 
        

