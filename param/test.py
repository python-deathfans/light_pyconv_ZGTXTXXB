from keras.layers import Input, Conv2D, MaxPooling2D,DepthwiseConv2D, SeparableConv2D
from keras.models import Model
from keras.utils import plot_model


inputs = Input(shape=(32, 32, 4))

# x = DepthwiseConv2D((5,1), padding='same', activation='relu')(inputs)
# x = DepthwiseConv2D((1,5), padding='same', activation='relu')(x)
# x = DepthwiseConv2D(5, padding='same', activation='relu')(inputs)

x = MaxPooling2D()(x)

outputs = Conv2D(64, 3, padding='same', activation='relu')(x)


model = Model(inputs=inputs, outputs=outputs)

model.summary()

plot_model(model, 'test.png', show_shapes=True)