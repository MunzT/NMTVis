import data

# Special tokens
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3
PAD_text = data.BLANK_WORD
SOS_text = data.BOS_WORD
UNK_text = "<UNK>"
EOS_text = data.EOS_WORD

# Vocabulary parameters
MIN_COUNT = 1
MIN_LENGTH = 3
# MAX_LENGTH = 40

# Training hyperparameters
teacher_forcing_ratio = 1
clip = 5.0
learning_rate = 0.0001
decoder_learning_ratio = 5

# Training logging parameters
eval_bleu_every_epochs = 1
print_loss_every_iters = 100
save_every_epochs = 1

# NN hyperparameters
hidden_size = 512
embed_size = 512
batch_size = 256
n_epochs = 25
n_layers = 2
attention = "general"
dropout = 0.1

prefix = ""
# Load vocabs from training files
load_vocabs = True
