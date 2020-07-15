import tensorflow as tf
import pickle
import os

class MyModel_1():
    def __init__(self, conv_input_size,
                 conv_kernel_size, conv_filters,
                 use_batch_norm = False, use_dropout = False):

        self.name = "My_model_1"
        self.conv_input_size = conv_input_size
        self.conv_kernel_size = conv_kernel_size
        self.conv_filters = conv_filters

        self.use_batch_nrom = use_batch_norm
        self.use_dropout = use_dropout

        self.n_layers_conv = len(conv_filters)

        self._build()

    def _build(self):
        ''' my model'''
        conv_input = tf.keras.layers.Input(shape=self.conv_input_size, name="conv_input")

        x = conv_input

        for i in range(self.n_layers_conv):
            conv_layer = tf.keras.layers.Conv2D(
                filters=self.conv_filters[i],
                kernel_size=self.conv_kernel_size[i],
                name="conv_" + str(i)
            )

            x = conv_layer(x)
            shape_before_flattening = tf.keras.backend.int_shape(x)[1:]

            if self.use_batch_nrom:
                x = tf.keras.layers.BatchNormalization()(x)
            if self.use_dropout:
                x = tf.keras.layers.Dropout(rate=0.25)(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=128, activation="relu")(x)
        conv_output = tf.keras.layers.Dense(units=10, activation="softmax")(x)

        self.conv_model = tf.keras.Model(conv_input, conv_output)

    def complile(self, lr):
        optimizer = tf.keras.optimizers.Adam(lr = lr)
        loss = "sparse_categorical_crossentropy"
        self.conv_model.compile(optimizer = optimizer, loss = loss, metrics = ["accuracy"])

    def save(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, "viz"))
            os.makedirs(os.path.join(folder, "weight"))

        with open(os.path.join(folder, "params.pkl"), "wb") as f:
            pickle.dump([
                self.conv_input_size
                , self.conv_filters
                , self.conv_kernel_size
                , self.use_batch_nrom
                , self.use_dropout
            ], f)

        self.plot_model(folder)

    def plot_model(self, run_folder):
        tf.keras.utils.plot_model(self.conv_model, to_file=os.path.join(run_folder, "viz/model.png"), show_shapes=True,
                                  show_layer_names=True)

    def load_weights(self, filepath):
        self.conv_model.load_weights(filepath)

    def train(self, x, y, batch_size, epochs, run_folder):
        self.conv_model.fit(x=x, y=y ,
                            batch_size= batch_size,
                            epochs=epochs,
                            validation_split=0.25)
















