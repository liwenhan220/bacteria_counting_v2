from cnn import *
import sys
import getopt

argv = sys.argv[1:]

try:
    opts, args = getopt.getopt(argv, "m:l:e:t:", 
                                ["model_dir=",
                                "log_dir=",
                                "num_epochs=",
                                "model_type="])
except:
    print("Error")

MODEL_DIR = "models"
LOG_DIR = "log"
EPOCHS = 10
MODEL_TYPE = "lenet"

for opt, arg in opts:
    if opt in ['-m', '--model_dir']:
        MODEL_DIR = arg

    elif opt in ['-l', '--log_dir']:
        LOG_DIR = arg

    elif opt in ['-e', '--num_epochs']:
        EPOCHS = int(arg)

    elif opt in ['-t', '--model_type']:
        MODEL_TYPE = arg

print('model dir: {}\nlog dir: {}\nnum epochs: {}\n'.format(MODEL_DIR, LOG_DIR, EPOCHS))

train_loop(EPOCHS, MODEL_DIR, LOG_DIR, MODEL_TYPE)