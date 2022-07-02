import os
import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

root_dir = os.path.normpath(os.path.join(os.getcwd(), os.pardir))
data_dir = os.path.normpath(os.path.join(root_dir, 'data'))
models_dir = os.path.normpath(os.path.join(root_dir, 'models'))
figures_dir = os.path.normpath(os.path.join(root_dir, 'reports', 'figures'))

class LinearDependency():
    def __init__(self, train_df, target_df, max_epochs=100):
        self.max_epochs = max_epochs
        self.train_df = train_df
        self.target_df = target_df

        self.weights_init = tf.keras.initializers.HeNormal(seed=123456789)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=1, use_bias=True, kernel_initializer=self.weights_init)
            ])

    def compile_and_fit(self, model, patience=2, model_name=None):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='mean_absolute_error',
                                                          patience=patience,
                                                          mode='min')

        model.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=tf.optimizers.Adam(),
                      metrics=[tf.metrics.MeanAbsoluteError()])

        history = model.fit(self.train_df, self.target_df, epochs=self.max_epochs,
                            #validation_data=()
                            callbacks=[early_stopping])

        tf.keras.models.save_model(model=model,
                                   filepath=os.path.join(
                                       root_dir, 'models', model_name),
                                   include_optimizer=True)

        return history

    def plot_weights(self):
        plt.bar(x = range(len(self.train_df.columns)),
        height=self.model.layers[0].kernel[:,0].numpy())
        axis = plt.gca()
        axis.set_xticks(range(len(self.train_df.columns)))
        _ = axis.set_xticklabels(self.train_df.columns, rotation=90)
        plt.show()


class Profiler():

    def __init__(self, ds):
        # Define local model directory
        self.model_dir = os.path.join(models_dir, 'Profiler')
        # Checkpoint location:
        self.checkpoint_dir = os.path.join(
            self.model_dir, 'training_checkpoints')
        self.weights_dir = os.path.join(self.model_dir, 'weights')
        self.ds = ds

    def classify(self, features: list, n_clusters=4):
        # Classify using k-means
        features = self.ds[features]
        k_means = KMeans(n_clusters=n_clusters).fit(features)
        identified_clusters = k_means.predict(features)
        identified_clusters = identified_clusters
        self.identified_clusters = identified_clusters
        try:
            self.ds.insert(0, 'class', self.identified_clusters)
        except:
            self.ds.drop('class', axis=1, inplace=True)
            self.ds.insert(0, 'class', self.identified_clusters)


    def plot(self, x, y, color=None, size=None):
        fig = px.scatter(self.ds, x=x, y=y, color=color,
                        log_x=True,
                        size=size,
                         template='plotly_white',
                         title="Profiles")
        fig.update_coloraxes(showscale=False)
        # fig.update_layout(xaxis_title="Popularity",)
        fig.show()
        #fig.write_html(os.path.join(figures_dir, 'profiler.html'))