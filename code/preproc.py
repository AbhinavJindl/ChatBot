import ast
import pickle
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt



'''
    Some constansts
'''
words_to_cons = 8000
sents_to_cons = 76800
max_len_sent = 30
min_len_sent = 5

'''
    This function is used to get all the conversations from movie_conversation file
    input : conv_file(name of the file)
'''
def get_conv(conv_file):
    ifile = open(conv_file)
    convs = ifile.readlines()
    ret_conv = []
    for line in convs:
        conv = line.split('+++$+++')[3].strip()
        conv = conv.replace('[','').replace(']','').replace('\'','').replace(' ','')
        conv_list = conv.split(',')
        ret_conv.append(conv_list)
    return ret_conv


'''
    This function returns a dictionary with keys as line numbers and values as lines.
    input : lines_file(name of the file)
'''
def get_lines(lines_file):
    ifile = open(lines_file)
    lines = ifile.readlines()
    ret_dict = {}

    for line in lines:
        line_list = line.split('+++$+++')
        line_num = line_list[0].strip()
        line = line_list[4]
#        line = re.sub(r'\.+',r' .',line)
#        line = re.sub(r'\?',r' ?',line)
#        line = re.sub(r',',r' ,',line)
        line = re.sub(r'\n',r' ',line)
        line = '<START> ' + line + ' <END>'
        ret_dict.update({line_num:line})
    return ret_dict

'''
    This function creates a tokenizer which has info of word2idx and idx2word
    input : tot_sent(list of all the sentences)
'''
def get_tokenizer(tot_sent):
    tokenizer = Tokenizer(num_words=words_to_cons,filters='!"#$%&\'()*+,-./:;=?@[\\]^_`{|}~',lower=True)
    tokenizer.fit_on_texts(tot_sent)
    return tokenizer

'''
    This function is used to sentences into index form(used for training).
    input : lines(sentences), sents(tags of sentences), tokenizer
'''
def get_text_tensor(lines,sents,tokenizer):
    sent = [lines[i] for i in sents]
    tensor = tokenizer.texts_to_sequences(sent)
    tensor = pad_sequences(tensor,padding='post', maxlen=max_len_sent)
    return tensor 

'''
    This function is used to extract questions from the conversations
    input : convs(conversations), lines(sentences)
'''
def info_of_dataset_cons(tot_sent):
    length_sents = [len(i) for i in tot_sent]
    # max_len = max(length_sents)
    # min_len = min(length_sents)
    # tot_bucktets = 10
    # spacing = (max_len - min_len)/tot_bucktets
    # hist = {}
    l = np.array(length_sents)
    

    





def get_ques_ans(convs, lines):
    questions = []
    answers = []
    for conv in convs:
        for i in range(len(conv)-1):
            x = len(lines[conv[i]].split(' '))
            y = len(lines[conv[i+1]].split(' '))
            if x <= max_len_sent and x >= min_len_sent and y <= max_len_sent and y >= min_len_sent:
                # print("addded qyes: ", lines[conv[i]])
                questions.append(conv[i])
                answers.append(conv[i+1])
                
    return questions,answers




if __name__ == '__main__':
    convs = get_conv('..\\dataset\\cornell_movie_dialogs_corpus\\movie_conversations.txt')
    lines = get_lines('..\\dataset\\cornell_movie_dialogs_corpus\\movie_lines.txt')
    tot_sent = [v for k,v in lines.items()]
    print(tot_sent)
    ques,ans = get_ques_ans(convs,lines)
    print('question and answers extracted.')
    print('creating tokenizer.......',end='')
    tokenizer = get_tokenizer(tot_sent)
    print('tokenization done.')
    print('preparing sequences for training........',end='')
    ques_tensor = get_text_tensor(lines,ques,tokenizer)
    ans_tensor = get_text_tensor(lines,ans,tokenizer)
    ques_tensor = ques_tensor[0:sents_to_cons]
    ans_tensor = ans_tensor[0:sents_to_cons]
    # for i in ques_tensor:
    #     print(i)
    train_ques, test_ques, train_ans, test_ans = train_test_split(ques_tensor,ans_tensor,test_size=0.2)
    print('sequences creation done.')
    print('saving the necassary data.....',end='')
    data_file = open('..\\saved_data\\data.pkl','wb')
    pickle.dump([train_ques,test_ques,train_ans,test_ans,tokenizer],data_file)
    data_file.close()
    print('data has been saved.')
    print('testing the correctness of the data saved....')
    print('Printing 5 sample question and answers pairs.')
    data_file = open('..\\saved_data\\data.pkl','rb')
    m_data = pickle.load(data_file)
    t = m_data[4]
    # for i in ques_tensor:
    #     print(i)
    ques_sample = t.sequences_to_texts(m_data[0])
    ans_sample = t.sequences_to_texts(m_data[2])
    for i in range(5):
        print(ques_sample[i] + ' : ' + ans_sample[i])


    
    