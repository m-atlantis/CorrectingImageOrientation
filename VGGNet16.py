from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten


class VGGNet16:
    @staticmethod
    def build(weights_path=None):
        vgg_model = Sequential()  # we define sequential function for building sequential CNN architecture.
        max_pool_2d = MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        vgg_model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
        vgg_model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))

        vgg_model.add(max_pool_2d)

        vgg_model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        vgg_model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
        vgg_model.add(max_pool_2d)

        vgg_model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        vgg_model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        vgg_model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
        vgg_model.add(max_pool_2d)

        vgg_model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        vgg_model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        vgg_model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        vgg_model.add(max_pool_2d)

        vgg_model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        vgg_model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        vgg_model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
        vgg_model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='vgg16'))

        vgg_model.add(Flatten(name='flatten'))

        # Here we used the ImageNet dataset trained VGG-16 model weight for using the transfer learning technique. And
        # we load the trained weights to our model.
        if weights_path:
            vgg_model.load_weights(weights_path)

        return vgg_model
