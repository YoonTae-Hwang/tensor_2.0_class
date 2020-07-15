import tensorflow as tf
import pickle
import os

class VGG():
    def __init__(self, conv_input_size,
                 conv_kernel_size, conv_filters,
                 conv_padding, conv_activation):

        self.name = "VGG"
        self.conv_input_size = conv_input_size
        self.conv_kernel_size = conv_kernel_size
        self.conv_filters = conv_filters
        self.conv_padding = conv_padding
        self.conv_activation = conv_activation

        self.n_layer_conv = len(conv_filters)

        self._build()

    def _build(self):

        conv_input = tf.keras.layers.Input(shape = self.conv_input_size, name = "VGG_input")

        x = conv_input

        for i in range(0, self.n_layer_conv // 2):
            conv_layer = tf.keras.layers.Conv2D(
                filters= self.conv_filters[i],
                kernel_size=self.conv_kernel_size[i],
                padding = self.conv_padding[i],
                activation=  self.conv_activation[i],
                name = "VGG_conv_" + str(i)
            )

            x  = conv_layer(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x)
        x = tf.keras.layers.Dropout(rate = 0.5)(x)

        for i in range(self.n_layer_conv // 2, self.n_layer_conv):
            conv_layer = tf.keras.layers.Conv2D(
                filters= self.conv_filters[i],
                kernel_size=self.conv_kernel_size[i],
                padding = self.conv_padding[i],
                activation=  self.conv_activation[i],
                name="VGG_conv_" + str(i)
            )

            x  = conv_layer(x)
        x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x)
        x = tf.keras.layers.Dropout(rate = 0.5)(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=512, activation="relu")(x)
        x = tf.keras.layers.Dropout(rate = 0.5)(x)
        x = tf.keras.layers.Dense(units=256, activation="relu")(x)
        x = tf.keras.layers.Dropout(rate = 0.5)(x)

        conv_output = tf.keras.layers.Dense(units=10, activation="softmax")(x)

        self.VGG = tf.keras.Model(conv_input, conv_output)

    def complie(self, lr):
        self.VGG.compile(optimizer = tf.keras.optimizers.Adam(),
                         loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

    def train(self, x, y, batch_size, epochs):

        self.VGG.fit(
            x,
            y,
            batch_size = batch_size,
            shuffle= True,
            epochs = epochs
        )

    def save(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, "viz"))
            os.makedirs(os.path.join(folder, "weights"))

        with open(os.path.join(folder, "params.pkl"), "wb") as f:
            pickle.dump([
                self.conv_input_size,
                self.conv_filters,
                self.conv_padding,
                self.conv_kernel_size,
                self.conv_activation
            ], f)

        self.plot_model(folder)

    def load_weights(self, file_path):
        self.VGG.load_weights(file_path)

    def plot_model(self, run_folder):
        tf.keras.utils.plot_model(self.VGG, to_file=os.path.join(run_folder, 'viz/model.png'), show_shapes=True,
                   show_layer_names=True)






