import glob
import random
import numpy as np
import pickle

#wikiextractorで取り出された記事からtab区切りの2文を要素とするコーパスを作る
#Nを記事数として
#train_X_N.txt train data
#valid_X_N.txt valid data

num_articles = 10000#記事の数
p = 0.8# 全体のpをTraining dataに当てます

train_data_txt = './data/_train_X_'+ str(num_articles) + '.txt'
valid_data_txt = './data/_valid_X_'+ str(num_articles) + '.txt'
train_title_txt = './data/_train_T_'+ str(num_articles) + '.txt'
valid_title_txt = './data/_valid_T_'+ str(num_articles) + '.txt'

titles = []#titleを保存するlist
concat_rows = []#2文を\tでつないだlist
c = 0#titleの番号


for directory in glob.glob('./extracted/*'):#sortedすると10000のときデータが約10倍に
    for name in glob.glob(directory+'/*'):
        with open(name, 'r') as r:
            for line in r:
                if c >= num_articles:
                    break
                #<docの削除、続く2行はtitleと空行なのでいらない
                if '<doc ' in line:
                    next(r)
                    next(r)
                    title = line[line.rfind("title=")+7:-3]
                #</docの削除、titleの番号を増やす
                elif '</doc>' in line:
                    c += 1
                    continue
                else:
                    # 空白・改行削除、大文字を小文字に変換、'. 'による文を分割
                    text = line.lower()
                    text = text.replace('.\n', '. ')
                    text = text.replace('."', '". ')
                    text = text.replace('.\'\'', '\'\'. ')
                    text = text.strip('\n')
                    text = text.split('. ')
                    #2文を\tでつなぐ ->段落の区切りを意識
                    max_text_len = (len(text) - 1) // 2
                    text = text[:max_text_len * 2]
                    for i in range(max_text_len):
                        concat_r = '\t'.join([text[2*i], text[2*i+1]])
                        concat_rows.append(concat_r)
                        titles.append(title)

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