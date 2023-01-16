# class to normalize activations: this is useful because it will stabilize training
# so activations have similar scale and distribution, preventing "exploding/vanishing"
# gradients, it will also help to converge faster, make the model more robust to unseen data 
# and its variations
from tensorflow.keras.layers import Input, Layer, Dropout,Conv2D, Activation, UpSampling2D, Add, Lambda
from tensorflow.keras.models import Model
class Normalize(Layer):
  # This line defines the Normalize class, which subclasses the Layer class. 
  # This makes the InstanceNormalization class a custom layer that can be used 
  # in a Keras model.
  # y = scale * (x - mean) / sqrt(variance + epsilon) + offset
  def __init__(self, **kwargs):
    #This line defines the __init__ method of the InstanceNormalization class. 
    #The __init__ method is called when an instance of the class is created, 
    #and it allows us to define any initialization logic for the instance. 
    #In this case, we pass any keyword arguments through to the parent class's 
    #__init__ method using the super function. The __init__ function of the 
    # SubClass calls the __init__ function of the BaseClass using the super 
    #function, which allows the SubClass to inherit the x variable from the 
    #BaseClass.
    super(Normalize, self).__init__(**kwargs)

  def build(self, input_shape):
    #build function creates more things you want for the model (perform initialization tasks and set any required attributes.)
    #, basically an extension of init
    self.scale = self.add_weight(name = "scale", shape=input_shape[-1:], initializer='ones', trainable = True)
    # This line creates a trainable weight called scale, which has shape input_shape[-1:] 
    # (i.e., it has the same number of elements as the number of channels in the input) 
    # and is initialized to all ones.
    self.offset = self.add_weight(name = "offset", shape = input_shape[-1:], initializer = 'zeros', trainable = True)
    #This line creates a trainable weight called offset, which has shape 
    # input_shape[-1:] (i.e., it has the same number of elements as the number 
    #of channels in the input) and is initialized to all zeros.
    super(Normalize, self).build(input_shape)
    #This is necessary to ensure that any initialization logic in the parent 
    #class's build method is run. It is generally a good idea to call the base 
    #class build function whenever you override it in a subclass, as this ensures 
    #that any initialization logic defined in the base class is properly executed. 

  def call(self, inputs, training = None):
    mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
    normalized = (inputs - mean) / tf.sqrt(variance + 1e-8)
    return self.scale * normalized + self.offset

#model
class Generator():
  def __init__(self, image_shape, num_gpu = 0):
        self.image_shape = image_shape
        self.num_gpu = num_gpu
  def resnet_blocks(self, input, filters, kernel_size=(3,3), strides=(1,1), use_dropout = False):
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   padding = 'same',
                   strides=strides)(input)
        x = Normalize()(x)
        x = Activation('relu')(x)

        if use_dropout:
            x = Dropout(0.5)(x)

        x = Conv2D(filters=filters,
                    kernel_size=kernel_size,
                    padding = 'same',
                    strides=strides,)(x)
        x = Normalize()(x)

        #skip connection
        skipped = Add()([input, x])
        return skipped

  def build_generator(self):
    """build generator architecture"""
    inputs = Input(shape = self.image_shape)
    x = Conv2D(filters=64, kernel_size=7, strides=1, padding='same')(inputs)
    #(batch_size, 413, 12, 64),
    x = Normalize()(x)
    x = Activation('relu')(x)
    # This layer applies a 2D convolution with 64 filters of size 7x7 and stride
    # 1 to the input tensor inputs, which has shape (batch_size, height, width, 
    # channels). The output tensor x has the same shape as inputs, because 
    # padding='same' specifies that the output should have the same size as the 
    # input.

    x = Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(x)
    #(batch size, 207, 6, 128)
    x = Normalize()(x)
    x = Activation('relu')(x)
    # x = Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(x): This
    # layer applies a 2D convolution with 128 filters of size 3x3 and stride 2 
    # to the input tensor x, which has shape (batch_size, height, width, 
    # channels). The output tensor x has shape (batch_size, height/2, width/2, 
    # 128), because the stride 2 reduces the size of the input by a factor of 2 
    # in each dimension (height and width).

    x = Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(x)
    #(batch size, 104, 3, 256)
    x = Normalize()(x)
    x = Activation('relu')(x)
    # The third Conv2D layer has filters=256, kernel_size=3, strides=2, and 
    # padding='same', so it applies a 2D convolution with 256 filters of size 
    # 3x3 and stride 2 to the input tensor. The output tensor has shape 
    # (batch_size, height/4, width/4, 256), because the stride 2 reduces the
    # size of the input by a factor of 2 in each dimension (height and width).

    # Apply 9 ResNet blocks, does not change the shape of the tensor.
    for i in range(9):
        x = self.resnet_blocks(x, 256, use_dropout=True)
    #(batch size, 104, 3, 256)

    x = UpSampling2D((2, 2))(x)
    #(batch size, 208, 6, 256)
    # The first UpSampling2D layer has size=(2, 2), so it upsamples the input 
    # tensor by a factor of 2 in each dimension (height and width). The output 
    # tensor has shape (batch_size, height/2, width/2, 256).

    x = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(x)
    #(batch size, 208, 6, 128)
    # The fourth Conv2D layer has filters=128, kernel_size=3, strides=1, and 
    # padding='same', so it applies a 2D convolution with 128 filters of size 
    # 3x3 and stride 1 to the input tensor. The output tensor has shape 
    # (batch_size, height/2, width/2, 128).
    x = Normalize()(x)
    x = Activation('relu')(x)

    x = UpSampling2D((2, 2))(x)
    #(batch size, 416, 12, 128)
    # The second UpSampling2D layer has size=(2, 2), so it upsamples the input
    # tensor by a factor of 2 in each dimension (height and width). The output 
    # tensor has shape (batch_size, height, width, 128).
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    #(batch size, 416, 12, 64)
    # The sixth Conv2D layer has filters=64, kernel_size=3, strides=1, and 
    # padding='same', so it applies a 2D convolution with 64 filters of size 
    # 3x3 and stride 1 to the input tensor. The output tensor has shape 
    # (batch_size, height, width, 64).
    x = Normalize()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=1, kernel_size=7, padding='same')(x)
    #(batch size, 416, 12, 1)
    # The final Conv2D layer has filters=1, kernel_size=7, and padding='same', 
    # so it applies a 2D convolution with 1 filter of size 7x7 and stride 1 to
    # the input tensor. The output tensor has shape (batch_size, height, width, 1).
    x = Activation('tanh')(x)

    # Add direct connection from input to output
    outputs = Add()([x, inputs])
    # outputs = Lambda(lambda z: z/2)(outputs)

    #UpSampling2D is just a simple scaling up of the image by using nearest neighbour or bilinear upsampling, so nothing smart. Advantage is it's cheap.
    #Conv2DTranspose is a convolution operation whose kernel is learnt (just like normal conv2d operation) while training your model. Using Conv2DTranspose will also upsample its input but the key difference is the model should learn what is the best upsampling for the job.
    # https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d
    #Conv2DTranspose can learn a set of weights that can be used to upsample the input tensor and generate a more detailed output tensor.
    #wrap together
    model = Model(inputs=inputs, outputs=outputs, name='Generator')
    if (self.num_gpu>1):
        model = multi_gpu_model(model, gpus=self.num_gpu)
    return model


gen = Generator((100, 84, 1)).build_generator()
#input has to have first,second dimension divisible by 4

430/4

#discriminator:
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU,Dense, Dropout,Flatten, Activation, UpSampling2D, Add, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
class Discriminator:
    def __init__(self, image_shape, num_gpu=1):
      #image shape is (height, width, channels)
        self.image_shape = image_shape
        self.num_gpu = num_gpu

    def build_discriminator(self):
        self.model = Sequential()
        self.model.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=self.image_shape))
        # the output shape will be (batch_size, new_image_shape[0]/2, new_image_shape[1]/2, 64)

        self.model.add(LeakyReLU(0.2))
        self.model.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2)))
        # (batch_size, new_image_shape[0]/4, new_image_shape[1]/4, 128), because the strides argument is set to (2, 2) and the padding argument is set to 'valid'.

        self.model.add(LeakyReLU(0.2))
        self.model.add(Flatten())
        # (batch_size, new_image_shape[0]/4*new_image_shape[1]/4*128)

        self.model.add(Dense(256))
        #(batch_size, 256), 256 neurons connected to everything in the previous layer

        self.model.add(LeakyReLU(0.2))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        #(batch_size, 1), 1 neurons connected to everything in the previous layer

        self.model.add(Activation('sigmoid'))

        d_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        self.model.compile(optimizer=d_opt, loss='binary_crossentropy')

        return self.model

        # if (num_gpu>1):
        #   self.model = multi_gpu_model(self.model, gpus=num_gpu)

disc = Discriminator((100, 84, 1)).build_discriminator()

def GAN(generator, discriminator, image_shape):

    discriminator.trainable = False

    inputs = Input(shape=image_shape)
    generated_images = generator(inputs)
    outputs = discriminator(generated_images)

    model = Model(inputs=inputs, outputs=[generated_images, outputs])

    gan_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    loss = ['binary_crossentropy']
    loss_weights = [1, 0.1]

    model.compile(optimizer=gan_opt, loss=loss, loss_weights=loss_weights)

    return model

def define_gan(g_model, d_model):
    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

gan = define_gan(gen, disc)
