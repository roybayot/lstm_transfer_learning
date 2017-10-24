# random seeds
SEEDS = [1337, 2451, 7721, 1291, 57]
# about the dataset
TRAIN_DATASET = "dataset/pan14-tweet-training.txt"
TEST_DATASET = "dataset/pan14-tweet-test.txt"
LANG = "english"
TASK = "gender"
OUT_DIR = "results/"

# about wordvectors
W2V_FILE = "embeddings/w2v-d300-skip-win5-en.model"
EMBEDDING_DIM = 300


# how much of training data is used as a dev set or validation set
VALIDATION_SPLIT = args.VALIDATION_SPLIT
 
 
# about the model
MAX_SEQUENCE_LENGTH = 50 
MAX_NB_WORDS = args.MAX_NB_WORDS 


N_EPOCHS = args.N_EPOCHS
BATCH_SIZE = args.BATCH_SIZE

VEC_TRAINABLE = args.VEC_TRAINABLE


N_FEATURES = args.N_FEATURES # controls num_features per filter for CNN
N_MEM_CELLS = 36 # controls how many tokens to keep in memory for LSTMS


DROPOUT_VAL = 0.1
REC_DROPOUT_VAL = 0.2 #recurrent dropout value
ACT_TYPE = "relu"

W_CONST_MAXN = args.W_CONST_MAXN 
W_L2_REG = args.W_L2_REG



