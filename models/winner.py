# main libs
import pandas as pd
import numpy as np
import json
import os
# data preparation libs
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from keras.utils import to_categorical
# model libs
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import BatchNormalization, LeakyReLU, Input,ReLU, Concatenate
from keras.optimizers import Adamax,SGD
from keras.regularizers import l1,l2
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import regularizers
import keras
import tensorflow as tf
from keras.models import Sequential, Model
# visualisation libs
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt

def make_or_replace_dir(directory):
  if not os.path.exists(directory):
    os.mkdir(directory)


class Winner:
  def __init__(self, experiment_name, model_path, c2v_name, data_path):
    self.name = experiment_name
    self.data_path = data_path
    self.model_path = f'{model_path}/model_saved/{experiment_name}'
    make_or_replace_dir(self.model_path)

    self.encoder_path = '/'.join([model_path, 'model_saved', c2v_name])
    
    self.import_matchs()
    self.import_encoding_infos()
    self.prepare_data()
    self.model = None


  
  def import_matchs(self):
    all_data = pd.read_csv(f'{self.data_path}/matchs/matchs.csv', sep=';', index_col=0)
    print(f'{len(all_data)} rows before dropna')
    all_data = all_data.dropna(axis=0, how='any')
    print(f'{len(all_data)} rows after dropna')
    self.winner = all_data.loc[:, [c for c in all_data.columns if 'champion' in c] + ['winner']]

    # which side is better ?
    # check score a dummy model predicting always majoritary side would have
    print('naive score by chsing best side', self.winner['winner'].value_counts()/len(self.winner))

  def import_encoding_infos(self):
    #encoding
    with open(f'{self.encoder_path}/champ_to_ID.txt') as json_file:
        dict_champ_to_ID = json.load(json_file)
        self.dict_champ_to_ID = dict([(key, int(value)) for key, value in dict_champ_to_ID.items()])
    with open(f'{self.encoder_path}/ID_to_champ.txt') as json_file:
        dict_ID_to_champ = json.load(json_file)
        self.dict_ID_to_champ = dict([(int(key), value) for key, value in dict_ID_to_champ.items()])

    self.N_champs = len(dict_ID_to_champ)

    # transform champ name to ID
    self.teams_ID = self.winner.drop('winner', axis=1).copy()
    for c in self.teams_ID.columns:
        self.teams_ID[c] = self.teams_ID[c].map(self.dict_champ_to_ID)


    # load encoder 2 times because having the same name is a problem for export
    self.encoderBlue = keras.models.load_model(f'{self.encoder_path}/champ2vec')
    self.encoderBlue._name = 'UNIQUENAMEFOREXPORT'
    for layer in self.encoderBlue.layers:
      layer.trainable = False
    self.encoderRed = keras.models.load_model(f'{self.encoder_path}/champ2vec')
    for layer in self.encoderRed.layers:
      layer.trainable = False

  def prepare_data(self):
    # encode each half
    # onehot each team
    blue = tf.keras.utils.to_categorical(self.teams_ID.iloc[:, range(5)], num_classes=self.N_champs).sum(axis=1)
    red = tf.keras.utils.to_categorical(self.teams_ID.iloc[:, range(5, 10)], num_classes=self.N_champs).sum(axis=1)


    map_color_id = {'blue': 1, 'red': 0}
    Y = self.winner['winner'].map(map_color_id)

    idx_train, idx_val = train_test_split(range(len(Y)))

    self.Xtrain_blue = blue[idx_train]
    self.Xval_blue = blue[idx_val]
    self.Xtrain_red = red[idx_train]
    self.Xval_red = red[idx_val]


    self.Ytrain = Y.iloc[idx_train]
    self.Yval = Y.iloc[idx_val]

  def add_model(self, model):
    self.model = model
    print(self.model.summary())

  def train(self, n_epochs):

    saved_model = f'{self.model_path}/tmp_winner.hdf5'
    mcp = ModelCheckpoint(saved_model, monitor="val_loss",
                          save_best_only=True, save_weights_only=False,
                        verbose=0)
    res = self.model.fit(
      x = [self.Xtrain_blue, self.Xtrain_red],
      y = self.Ytrain,
      batch_size = 512,
      epochs = n_epochs,
      callbacks = [mcp],
      validation_data = ([self.Xval_blue, self.Xval_red], self.Yval),
      verbose=1
    )

    for key, value in res.history.items():
        plt.plot(value, label=key)
        plt.legend()
    plt.show()

    # load best mode"l exported with callback
    self.model.load_weights(saved_model)
    make_or_replace_dir(f'{self.model_path}/winner')
    self.model.save(f'{self.model_path}/winner/')




