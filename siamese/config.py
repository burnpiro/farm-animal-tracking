from easydict import EasyDict

__C = EasyDict()

cfg = __C

# create NN dict
__C.NN = EasyDict()

__C.NN.INPUT_SIZE = 224  # 224, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608
__C.NN.GRID_SIZE = 7  # the same as output layer from the model
__C.NN.ALPHA = 0.75  # one of mobilenet alphas 0.35 0.5 0.75 1.0 1.3 1.4
__C.NN.IOU_LOSS_THRESH = 0.5

# create Train options dict
__C.TRAIN = EasyDict()
__C.TRAIN.DATA_PATH = "./data/WIDER_train/images/"
__C.TRAIN.ANNOTATION_PATH = "./data/wider_face_split/wider_face_train_bbx_gt.txt"
__C.TRAIN.PATIENCE = 30  # for EarlyStopping
__C.TRAIN.EPOCHS = 150
__C.TRAIN.BATCH_SIZE = 16
__C.TRAIN.LEARNING_RATE = 1e-4
__C.TRAIN.WEIGHT_DECAY = 0.0005
__C.TRAIN.LR_DECAY = 0.0001

# create TEST options dict
__C.TEST = EasyDict()
__C.TEST.DATA_PATH = "./data/WIDER_val/images/"
__C.TEST.ANNOTATION_PATH = "./data/wider_face_split/wider_face_val_bbx_gt.txt"
__C.TEST.DATA_AUG = False
__C.TEST.DETECTED_IMAGE_PATH = "./output/detection/"