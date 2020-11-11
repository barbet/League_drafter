# main libs
import os
import pandas as pd
import numpy as np
import json
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
from DL_utilities import Generator_amputor_team

def make_or_replace_dir(directory):
  if not os.path.exists(directory):
    os.mkdir(directory)

class Champ2Vec:
  def __init__(self, experiment_name, model_path, data_path, params_generator):
    self.name = experiment_name
    self.data_path = data_path
    self.model_path = f'{model_path}/model_saved/{experiment_name}'
    make_or_replace_dir(self.model_path)

    self.import_matchs()
    self.create_champions_ID()
    self.split(params_generator)
    self.model = None
    self.champ_to_encoding=None

  def import_matchs(self):

    all_data = pd.read_csv(f'{self.data_path}/matchs/matchs.csv', sep=';', index_col=0)
    all_data = all_data.dropna(axis=0, how='any')
    champions = all_data.loc[:, [c for c in all_data.columns if 'champion' in c]]
    
    
    # split 5c5 into teams format: only 5 champions
    # split each row containing both teams
    # to two rows
    teams = pd.concat([
            champions.loc[:, [c for c in champions.columns if 'red' in c]].rename(
                columns=dict(zip([c for c in champions.columns if 'red' in c], range(5)))
            ),
            champions.loc[:, [c for c in champions.columns if 'blue' in c]].rename(
                columns=dict(zip([c for c in champions.columns if 'blue' in c], range(5)))
            )
        ]
        , axis=0
    )
    print(f'{len(teams)} teams in dataset')
    self.teams = teams

  def create_champions_ID(self):
    # create a unique ID for each champion
    _, uniques = self.teams.unstack().factorize()
    # to have a consistent encoding
    uniques = sorted(uniques)
    self.dict_champ_to_id = dict(zip(uniques, range(len(uniques))))
    self.dict_id_to_champ = dict(zip(range(len(uniques)), uniques))
    self.N_champs = len(uniques)
    print(f'{self.N_champs} different champions')
    # export dicts
    with open(f'{self.model_path}/ID_to_champ.txt', 'w') as outfile:
      json.dump(self.dict_id_to_champ, outfile)
    with open(f'{self.model_path}/champ_to_ID.txt', 'w') as outfile:
      json.dump(self.dict_champ_to_id, outfile)

    # initial data seen as ID
    self.teams_ID = self.teams.copy()
    for c in self.teams_ID.columns:
        self.teams_ID[c] = self.teams_ID[c].map(self.dict_champ_to_id)


  def composition_counter(self):
    # most seen team compositions
    teams = self.teams.copy()
    teams.values.sort()
    teams['counter'] = 1
    res = teams.groupby([0,1,2,3,4])['counter'].sum()

    return res.sort_values(ascending=False).head(5)


  def split(self, params_generator):
    X = self.teams_ID.copy() # ?.drop_duplicates()
    idx_train, idx_val = train_test_split(X.index)
    Xtrain = X.loc[idx_train]
    Xval = X.loc[idx_val]
    self.training_generator = Generator_amputor_team(Xtrain, self.N_champs, **params_generator) 
    self.validation_generator = Generator_amputor_team(Xval, self.N_champs, **params_generator)

  def add_model(self, model, encoder):
    """
    model compiled
    """
    self.model = model
    self.encoder=encoder
    print(model.summary())

  def train(self, n_epoch):
    if self.model is None:
      print('need to add_model')
      return

    saved_model = f'{self.model_path}/tmp_champ2vec.hdf5'
    mcp = ModelCheckpoint(saved_model, monitor="val_loss",
                          save_best_only=True, save_weights_only=False,
                        verbose=0)

    # Train model on dataset
    res = self.model.fit_generator(
      generator=self.training_generator,
      validation_data=self.validation_generator,
      use_multiprocessing=False,
      epochs = n_epoch,
      callbacks=[mcp],
      workers=30,
      verbose=2)

    # visualize training
    for key, value in res.history.items():
        plt.plot(value, label=key)
        plt.legend()
    plt.show()

    # load best mode"l exported with callback
    self.model.load_weights(saved_model)
    make_or_replace_dir(f'{self.model_path}/champ2vec')
    self.encoder.save(f'{self.model_path}/champ2vec/')

  

  def show_preds(self):
    # test prediction
    example = self.teams_ID.loc[np.random.choice(
      self.teams_ID.index
      ), :]

    for idx, t in example.iterrows():
        for col in example.columns:
            x = t.drop(col)
            x_onehot = tf.keras.utils.to_categorical(x, num_classes=self.N_champs).sum(axis=0)
            y = t[col]

            pred = self.model.predict(x_onehot.reshape(1,-1))[0]

            print('-------------------')
            print('team', apply_text(x.to_frame(), self.dict_id_to_champ))
            print('removed', self.dict_id_to_champ[y])
            print('found', apply_text(pd.DataFrame(np.argsort(-pred)[:5]), self.dict_id_to_champ))
        break

  def encoding_analysis(self):
    # encode each champion
    champ_to_encoding = pd.DataFrame(columns=range(self.encoder.output.shape[1]))
    for champ_name, champ_id in self.dict_champ_to_id.items():
        onehot = tf.keras.utils.to_categorical(np.array([champ_id]), num_classes=self.N_champs)
        champ_to_encoding.loc[champ_name] = self.encoder.predict(onehot.reshape(1,-1))[0]
    self.champ_to_encoding = champ_to_encoding


    fig_path=f'{self.model_path}/figs'
    make_or_replace_dir(fig_path)    
    # 0 in imread cause reasons..
    # https://stackoverflow.com/questions/39379609/matplotlib-valueerror-invalid-png-header
    all_images = [
        plt.imread(f'{self.data_path}/images/loading/{c}.png', 0) for c in self.dict_champ_to_id.keys()
    ]
    zoom_same = 0.25
    zoom_default = 0.3

    for axis in range(self.encoder.output.shape[1]):
        for other_axis in range(axis, self.encoder.output.shape[1]):
            fig, ax = plt.subplots(figsize=(30,30))
            z = zoom_same if (axis==other_axis) else zoom_default
            imscatter(
              self.champ_to_encoding[axis],
              self.champ_to_encoding[other_axis],
              all_images, zoom=z, ax=ax, dither_axis=(axis==other_axis)
            )
            plt.title(f'show axis {axis} and {other_axis}')
            plt.savefig(f'{fig_path}/{axis}_{other_axis}.png')
            plt.close(fig)


def apply_text(df, dict_id_to_champ):
  """
  utility to transform rows of ID to rows of champ names
  """
  return df.applymap(lambda id: dict_id_to_champ[id])

def imscatter(x, y, images, ax=None, zoom=1, dither_axis=False):
  """
  # from https://stackoverflow.com/questions/22566284/matplotlib-how-to-plot-images-instead-of-points
  """
  if ax is None:
      ax = plt.gca()

  if dither_axis:
      y = np.array([v + (idx%5 - 2)*0.3 for idx,v in enumerate(y)])
  x, y = np.atleast_1d(x, y)
  artists = []
  for x0, y0, image in zip(x, y, images):
      im =  OffsetImage(image, zoom=zoom)
      ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
      artists.append(ax.add_artist(ab))
  ax.update_datalim(np.column_stack([x, y]))
  ax.autoscale()
  return artists