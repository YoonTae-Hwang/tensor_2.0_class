from utils.loaders import load_mnist
from models.VGG import VGG
import os


#실행 매게변수
SECTION = "VGG"
RUN_ID = "0001"
DATA_NAME = "F_MNIST"
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

os.environ["PATH"] += os.pathsep + "C:\\Program Files (x86)\\Graphviz2.38\\bin\\"

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))


MODE =  'build' #'load' #
(x_train, y_train), (x_test, y_test) = load_mnist()



VGG_model = VGG(
    conv_input_size = (28,28,1)
    , conv_kernel_size = [3,3,3,3]
    , conv_filters = [32,64,128,256]
    , conv_padding = ["same","same","same","valid"]
    , conv_activation = ["relu","relu","relu","relu"]
)

if MODE == 'build':
    VGG_model.save(RUN_FOLDER)
else:
    VGG_model.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))



#train
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
INITIAL_EPOCH = 0
VGG_model.complie(LEARNING_RATE)

VGG_model.train(
    x = x_train,
    y = y_train,
    batch_size= BATCH_SIZE,
    epochs = 1
)