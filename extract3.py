import glob
import random
from collections import Counter, OrderedDict
import pickle
import numpy as np
import os

min_words = 5000
max_words = 10000
num_articles = 10000
min_len = 100
p = 0.8

dir_name = str(num_articles) + '_' + str(min_words) + '_' + str(max_words) + '_' + str(min_len)
if not os.path.isdir('./data1/'+dir_name):
    os.makedirs('./data1/'+dir_name)

train_data_txt = './data1/'+dir_name+'/train_X_'+dir_name# + '.txt'
valid_data_txt = './data1/'+dir_name+'/valid_X_'+dir_name# + '.txt'
train_title_txt = './data1/'+dir_name+'/train_T_'+dir_name# + '.txt'
valid_title_txt = './data1/'+dir_name+'/valid_T_'+dir_name# + '.txt'
pt_train_data_txt = './data1/'+dir_name+'/pt_train_X_'+dir_name# + '.txt'
pt_valid_data_txt = './data1/'+dir_name+'/pt_valid_X_'+dir_name# + '.txt'
pt_train_title_txt = './data1/'+dir_name+'/pt_train_T_'+dir_name# + '.txt'
pt_valid_title_txt = './data1/'+dir_name+'/pt_valid_T_'+dir_name# + '.txt'
p_list_pkl = './data1/'+dir_name+'/p_list'+dir_name+'.pkl'


def for_corpus(words):
    words = words.replace('."', '". ')
    words = words.replace('.\'\'', '\'\'. ')
    words = words.replace('. ', ' . ')
    words = words.replace('.\n', ' . ')
    words = words.replace("'", " ' ")
    words = words.replace('"', ' " ')
    words = words.replace(", ", " , ")
    words = words.replace(":", " : ")
    words = words.replace(";", " ; ")
    words = words.replace("(", " ( ")
    words = words.replace(")", " ) ")
    words = words.replace("[", " [ ")
    words = words.replace("]", " ] ")
    words = words.replace("?", " ? ")
    words = words.replace("!", " ! ")
    words = words.split()
    return words

with open('title_num_words', 'r') as f:
    article_list = [line[:line.find('\t')] for line in f if min_words < int(line[line.find('\t'):-1]) < max_words]

random.shuffle(article_list)
if num_articles < len(article_list):
    article_list = article_list[:num_articles]
else:
    print('num_articles =', num_articles)
    print('but only', len(article_list), 'articles exist')

titles = []#titleを保存するlist
concat_rows = []#偶数個の文を\tでつないだlist
texts = []#全文を単語に分割して保存したlist
count_list = {}#全文の単語のcounter
for article in article_list:
    f = open(glob.glob('./articles/*/'+article+'.txt')[0], 'r')
    text = f.read()
    f.close()
    text = for_corpus(text)
    texts += text
    count_list[article] = Counter(text)

    l_period = [i+1 for i, x in enumerate(text) if x == '.']
    #l_period = [0] + l_period
    l_periods = l_period[1::2]
    i0 = 0
    l_sep = [0]
    ll = [0]
    for j, i in enumerate(l_periods):
        if i - i0 > min_len:
            l_sep.append(2*(j+1))
            ll.append(i)
            i0 = i
    l_period = [0] + l_period
    sentence_list = [' '.join(text[l_period[i]:l_period[i+1]]) for i in range(len(l_period)-1)]
    concat_row = ['\t'.join(sentence_list[l_sep[i]:l_sep[i+1]]) for i in range(len(l_sep)-1)]
    title = [article for _ in range(len(l_sep)-1)]
    concat_rows += concat_row
    titles += title

num_inputs = len(concat_rows)
print('num_inputs : ', num_inputs)
print('train_ratio : ', p)
data_label = np.arange(num_inputs)
random.shuffle(data_label)
train_label = data_label[:int(num_inputs*p)]
valid_label = data_label[int(num_inputs*p):]

with open(train_data_txt, 'w') as f:
    for i in train_label:
        f.write(concat_rows[i]+'\n')
with open(valid_data_txt, 'w') as f:
    for i in valid_label:
        f.write(concat_rows[i]+'\n')
with open(train_title_txt, 'w') as f:
    for i in train_label:
        f.write(titles[i]+'\n')
with open(valid_title_txt, 'w') as f:
    for i in valid_label:
        f.write(titles[i]+'\n')

def for_pt(concat_rows, titles, label):
    pt_concat_rows, pt_titles = [], []
    for i in label:
        sentence = concat_rows[i].split('\t')
        pt_concat_row = ['\t'.join([sentence[2*j], sentence[2*j+1]]) for j in range(int(len(sentence)/2))]
        pt_title = [titles[i] for _ in range(int(len(sentence)/2))]
        pt_concat_rows += pt_concat_row
        pt_titles += pt_title
    return pt_concat_rows, pt_titles

pt_train_concat_rows, pt_train_titles = for_pt(concat_rows, titles, train_label)
pt_valid_concat_rows, pt_valid_titles = for_pt(concat_rows, titles, valid_label)

with open(pt_train_data_txt, 'w') as f:
    for r in pt_train_concat_rows:
        f.write(r+'\n')
with open(pt_valid_data_txt, 'w') as f:
    for r in pt_valid_concat_rows:
        f.write(r+'\n')
with open(pt_train_title_txt, 'w') as f:
    for t in pt_train_titles:
        f.write(t+'\n')
with open(pt_valid_title_txt, 'w') as f:
    for t in pt_valid_titles:
        f.write(t+'\n')

total_count_list = Counter(texts)
#p_list = {}
p_list = OrderedDict()
for article in article_list:
    p = {}
    for word in count_list[article]:
        p[word] = count_list[article][word] / total_count_list[word]
    p_list[article] = p

with open(p_list_pkl, 'wb') as f:
    pickle.dump(p_list, f)
#for article in article_list:
#    words_count = 0
#    freq_words_count = 0
#    freq_words = 0
#    for word in count_list[article]:
#        words_count += count_list[article][word]
#        if count_list[article][word] / total_count_list[word] == 1:
#            #print(article, word, count_list[article][word] / total_count_list[word])
#            freq_words_count += count_list[article][word]
#            freq_words += 1
#    print(freq_words_count / words_count, freq_words / len(count_list[article]))