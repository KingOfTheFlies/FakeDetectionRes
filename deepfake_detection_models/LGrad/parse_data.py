# from glob import glob
# import re

# path = '../../datasets'
# files = glob(path + '/*/*/*', recursive = True)

# files = [ re.sub(path+'/', '', re.sub('/1_fake', '', f)) for f in files]
# files = [ re.sub(path+'/', '', re.sub('/0_real', '', f)) for f in files]

# files = list(set(files))
# print(' '.join(files))

import tensorflow as tf

print("TensorFlow version:", tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print("GPUs Available: ", len(gpus))
    for gpu in gpus:
        print(" -", gpu)
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPUs found")
# print("PyTorch version:", torch.__version__)
# print("Num GPUs Available in PyTorch: ", torch.cuda.device_count())