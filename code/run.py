''''
    This file consists of model with embedding layer.
    Embedding layer is added later. After testing we found adding gives better results.
'''



from keras.layers import Input,Embedding, LSTM,Dense
from keras import Model
from keras.callbacks import ModelCheckpoint
import pickle
import numpy as np
from keras.utils import to_categorical
from numpy import array
import sys
import matplotlib.pyplot as plt
#import tensorflow as tf


#parameters
num_encoder_tokens = 8000
latent_dim = 50
num_decoder_tokens = 8000
max_size_of_sent = 30
batch_size = 256
vocab = 8000
units=10
epochs = 50

#generator function
#this function is used as a generator as the data is huge.
def get_dataset(ques,ans,batch_size=batch_size):
    
    
    encoder_input_data = np.zeros(shape=(batch_size,max_size_of_sent),dtype='float32')
    decoder_input_data = np.zeros(shape=(batch_size,max_size_of_sent),dtype='float32')
    decoder_target_data = np.zeros(shape=(batch_size,max_size_of_sent,num_decoder_tokens))
    
    
    #Here we can see variable t as timestamp
    while True:
        for j in range(0, len(ques), batch_size):
            x = min(len(ques)-j,batch_size)
            encoder_input_data = np.zeros((x, max_size_of_sent),dtype='float32')
            decoder_input_data = np.zeros((x, max_size_of_sent),dtype='float32')
            decoder_target_data = np.zeros((x, max_size_of_sent, num_decoder_tokens),dtype='float32')
            for i in range(x):
                for t in range(max_size_of_sent):
                    encoder_input_data[i, t] = ques[i+j][t] # encoder input seq
                for t in range(max_size_of_sent):
                    decoder_input_data[i, t] = ans[i+j][t] # decoder input seq
                    if t>0:
                        #This is teacher forcing. The input at timestep 't' is forced to be the output at timestep 't-1'.
                        #This is the output of the dense layer in the decoder part of model. It is a one hot encoding.
                        decoder_target_data[i, t - 1,ans[i+j][t]]  = 1.0
            yield([encoder_input_data, decoder_input_data], decoder_target_data)

#%%
# Encoder architecture starts here
encoder_inputs = Input(shape=(None,))
enc_emb =  Embedding(num_encoder_tokens, latent_dim, mask_zero = True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
#encoder states are passed forward to decoder.
encoder_states = [state_h, state_c]

# Decoder architecture starts here
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(num_decoder_tokens, latent_dim, mask_zero = True)
dec_emb = dec_emb_layer(decoder_inputs)
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['acc'])

#inference models.
#encoder model
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder setup
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2= dec_emb_layer(decoder_inputs) # Get the embeddings of the decoder sequence

# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2) # A dense softmax layer to generate prob dist. over the target vocabulary

# Final decoder model
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)

#given a list of questions, It outputs output sentences
def response(list_ques):
    pred = []
    for i in list_ques:
        seq = tokenizer.texts_to_sequences([i])
        seq = np.pad(seq, ((0,0),(0,30)), mode='constant')
        gen_func = get_dataset(batch_size=1, ques=seq, ans=seq )
        inp = next(gen_func)[0][0]
        predicted = decode_sequence(inp, encoder_model, decoder_model, tokenizer)
        pred.append(predicted)
    return pred

def chat():
    while(1):
        print('question : ',end='')
        ques = input()
        ques = ques.lower()
        pred = response(['<start> ' + ques + ' <end>'])[0]
        print('answer : ',pred)

def decode_sequence(input_seq,encoder_model,decoder_model,tokenizer):

    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = tokenizer.texts_to_sequences(['<start>'])[0][0]
    
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
#        print(output_tokens,h,c)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = tokenizer.sequences_to_texts([[sampled_token_index]])[0]
        decoded_sentence += ' '+sampled_char
    
        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '<end>' or
           len(decoded_sentence) > max_size_of_sent):
            stop_condition = True
    
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index
    
        # Update states
        states_value = [h, c]
    
    return decoded_sentence
#%%

#https://datascience.stackexchange.com/questions/34444/what-is-the-difference-between-fit-and-fit-generator-in-keras
#get_dataset()
#%%
ifile = open('../saved_data/data.pkl','rb')
m_data = pickle.load(ifile)
train_ques = m_data[0]
train_ans = m_data[2]
test_ques = m_data[1]
test_ans = m_data[3]
tokenizer = m_data[4]
#%%


filepath = "weights_emb_layer.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max',period=2)
callbacks_list = [checkpoint]

train_mode = 0
# To fit the model given the dataset.
if len(sys.argv) == 2:
    if sys.argv[1] == 'trainmode':
        train_mode = 1
    elif sys.argv[1] == 'chatmode':
        train_mode = 0
    else:
        sys.exit('Wrong arguments!!')
if train_mode == 1:
    history = model.fit_generator(generator=get_dataset(train_ques,train_ans),
                    steps_per_epoch=len(train_ques)//batch_size,
                    callbacks=callbacks_list,
                    validation_data=get_dataset(test_ques,test_ans),
                    validation_steps=len(test_ques)//batch_size,epochs=epochs)
    #this will give info of 
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

model.load_weights(filepath='weights_emb_layer.hdf5')

#This function is used to chat with the bot.


chat()    
    
    

