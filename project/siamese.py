import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import keras.backend as K

def ConvBlock(x, filters, kernel_size):
    
    x = tf.keras.layers.Conv2D(filters = filters, 
                           kernel_size = kernel_size, 
                           activation = 'relu',
                           )(x)

    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    return x

def FullyConnected(x):
    x = tf.keras.layers.Conv2D(256, 3, activation = 'relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation = 'relu')(x)
    
    return x

def BaseNetwork(x):
    
    
    x = ConvBlock(x, filters = 64, kernel_size = 10)
    x = ConvBlock(x, filters = 128, kernel_size = 7)
    x = ConvBlock(x, filters = 256, kernel_size = 7)
    
    
    x = FullyConnected(x)
    
    return x
    
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


class MyContrastiveLoss(tf.keras.losses.Loss):
    
    def __init__(self, margin = 1, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        
    def call(self, y_true, y_pred):
        sqr_pred = K.square(y_pred)
        margin_sqr = K.square(K.maximum(self.margin - (y_pred), 0))
        return K.mean((1 - y_true) * sqr_pred + (y_true) * margin_sqr)
    
    
def build_model():
    
    inputs = tf.keras.layers.Input(shape = (100, 100, 3))
    embedding_network = tf.keras.Model(inputs, BaseNetwork(inputs), name = 'embedding_network')
    
    input_top = tf.keras.layers.Input(shape = (100,100,3), name = 'input_top')
    vect_top = embedding_network(input_top)

    input_bottom = tf.keras.layers.Input(shape = (100,100,3), name = 'input_bottom')
    vect_bottom = embedding_network(input_bottom)

    eucli_distance = tf.keras.layers.Lambda(euclidean_distance, output_shape = (1,))(
        [vect_top, vect_bottom]
    )
    normal_layer = tf.keras.layers.BatchNormalization()(eucli_distance)
    output = tf.keras.layers.Dense(1, activation = 'sigmoid')(normal_layer)
    
    model = tf.keras.Model(inputs = [input_top, input_bottom], outputs = output, name = 'siamese_network')
    
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)
    loss = MyContrastiveLoss(1)
    model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy'])
    
    return model