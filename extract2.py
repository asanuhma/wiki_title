import glob
import random
import numpy as np
import pickle

#wikiextractorで取り出された記事からtab区切りの2文を要素とするコーパスを作る
#Nを記事数として
#train_X_N.txt train data
#valid_X_N.txt valid data
num_words_list = np.loadtxt('num_words')

def for_corpus(words):
    words = words.replace('."', '". ')
    words = words.replace('.\'\'', '\'\'. ')
    words = words.replace('. ', ' . ')
    words = words.replace('.\n', ' . ')
    words = words.replace("'", " ' ")
    words = words.replace('"', ' " ')
    words = words.replace(",", " , ")
    words = words.replace(":", " : ")
    words = words.replace(";", " ; ")
    words = words.replace("(", " ( ")
    words = words.replace(")", " ) ")
    words = words.strip('\n')
    words = words.split()
    return words

num_articles = 100#記事の数
min_words = 5000
max_words = 10000
min_len = 100
p = 0.8# 全体のpをTraining dataに当てます

train_data_txt = './data/_train_X_'+ str(num_articles) + '_' + str(min_words) + '_' + str(max_words) + '_' + str(min_len) + '.txt'
valid_data_txt = './data/_valid_X_'+ str(num_articles) + '_' + str(min_words) + '_' + str(max_words) + '_' + str(min_len) + '.txt'
train_title_txt = './data/_train_T_'+ str(num_articles) + '_' + str(min_words) + '_' + str(max_words) + '_' + str(min_len) + '.txt'
valid_title_txt = './data/_valid_T_'+ str(num_articles) + '_' + str(min_words) + '_' + str(max_words) + '_' + str(min_len) + '.txt'

titles = []#titleを保存するlist
concat_rows = []#2文を\tでつないだlist
c = 0#titleの番号
C = 0

for directory in sorted(glob.glob('./extracted/*')):#sortedすると10000のときデータが約10倍に
    for name in sorted(glob.glob(directory+'/*')):
        with open(name, 'r') as r:
            for line in r:
                if C >= num_articles:
                    break
                #<docの削除、続く2行はtitleと空行なのでいらない
                if '<doc ' in line:
                    next(r)
                    next(r)
                    if min_words < num_words_list[c] < max_words:
                        title = line[line.rfind("title=")+7:-3]
                        texts = []
                #</docの削除、titleの番号を増やす
                elif '</doc>' in line:
                    if min_words < num_words_list[c] < max_words:
                        l = [i for i, x in enumerate(texts) if x == '.']
                        i0 = 0
                        L = [0]
                        for i in l:
                            if i - i0 > min_len:
                                L.append(i+1)
                                i0 = i
                        texts = [' '.join(texts[L[i]:L[i+1]]) for i in range(len(L)-1)]
                        max_text_len = (len(texts) - 1) // 2
                        texts = texts[:max_text_len * 2]
                        for i in range(max_text_len):
                            concat_r = '\t'.join([texts[2*i], texts[2*i+1]])
                            concat_rows.append(concat_r)
                            titles.append(title)
                        C += 1
                    c += 1
                else:
                    # 空白・改行削除、大文字を小文字に変換、'. 'による文を分割
                    if min_words < num_words_list[c] < max_words:
                        text = line.lower()
                        text = for_corpus(text)
                        texts = texts + text
                    #2文を\tでつなぐ ->段落の区切りを意識
                    #max_text_len = (len(text) - 1) // 2
                    #text = text[:max_text_len * 2]
                    #for i in range(max_text_len):
                    #    concat_r = '\t'.join([text[2*i], text[2*i+1]])
                    #    concat_rows.append(concat_r)
                    #    titles.append(title)

num_line_pairs = len(concat_rows)
print('num_line_pairs : ', num_line_pairs)
print('train_ratio : ', p)
data_label = np.arange(num_line_pairs)
random.shuffle(data_label)
train_label = data_label[:int(num_line_pairs*p)]
valid_label = data_label[int(num_line_pairs*p):]

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

print('num_train_line_pairs : ', len(train_label))
print('num_valid_line_pairs : ', len(valid_label))