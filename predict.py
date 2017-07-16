# --------------------------------------------------------------------------------------------
# import statements
# --------------------------------------------------------------------------------------------
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
from keras.datasets import imdb
from keras.preprocessing import sequence
from gensim.models import word2vec
from os.path import join, exists, split
from keras.models import load_model
np.random.seed(0)


# --------------------------------------------------------------------------------------------
# setting the parameters
# --------------------------------------------------------------------------------------------
# Model Hyperparameters
embedding_dim = 50
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50
# Training parameters
batch_size = 64
num_epochs = 5
# Prepossessing parameters
sequence_length = 400
max_words = 5000
# Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 10
# length of the vocab on which the model is trained on
vocabLen = 88585



# --------------------------------------------------------------------------------------------
# loading the existing word2vec model
# --------------------------------------------------------------------------------------------
model_dir = 'models'
model_name = "{:d}features_{:d}minwords_{:d}context".format(embedding_dim, min_word_count, context)
model_name = join(model_dir, model_name)
if exists(model_name):
	embedding_model = word2vec.Word2Vec.load(model_name)
	print('Loading existing Word2Vec model \'%s\'' % split(model_name)[-1])
else:
	print("first train the Word2Vec model")


# --------------------------------------------------------------------------------------------
# loading the main neural network model
# --------------------------------------------------------------------------------------------
if exists('sentimentAnalysis-weights.h5'):
	model = load_model('sentimentAnalysis-weights.h5')
else:
	input_shape = (sequence_length,)
	model_input = Input(shape=input_shape)

	# Static model does not have embedding layer
	z = Embedding(vocabLen, embedding_dim, input_length=sequence_length, name="embedding")(model_input)

	z = Dropout(dropout_prob[0])(z)

	# Convolutional block
	conv_blocks = []
	for sz in filter_sizes:
	    conv = Convolution1D(filters=num_filters,
	                         kernel_size=sz,
	                         padding="valid",
	                         activation="relu",
	                         strides=1)(z)
	    conv = MaxPooling1D(pool_size=2)(conv)
	    conv = Flatten()(conv)
	    conv_blocks.append(conv)
	z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

	z = Dropout(dropout_prob[1])(z)
	z = Dense(hidden_dims, activation="relu")(z)
	model_output = Dense(1, activation="sigmoid")(z)

	model = Model(model_input, model_output)
	model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# --------------------------------------------------------------------------------------------
# printing the model summary
# --------------------------------------------------------------------------------------------
model.summary()


# --------------------------------------------------------------------------------------------
# input to the model to predict
# --------------------------------------------------------------------------------------------
sentence_to_be_classified = input("enter the sentence")
vocabulary = imdb.get_word_index()

inputToTheModel = [vocabulary[word] for word in sentence_to_be_classified.split()]

for i in range(400-len(inputToTheModel)):
	inputToTheModel.append(0)

print("the input to the model is ",inputToTheModel)

prediction = model.predict(zz)[0]
print(prediction)
print('I think the sentiment of the entered statement is ' + ('positive' if prediction > 0.5 else 'negative') + ' with ' + str(prediction * 100) + '% confidence.')
