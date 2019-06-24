import glob
import random

#wikiextractorで取り出された記事からtab区切りの2文を要素とするコーパスを作る
#Nを記事数として
#train_X_N.txt train data
#valid_X_N.txt valid data

num_articles = 10000#記事の数
p = 0.8# 全体のpをTraining dataに当てます

filename = 'tmp_'+ str(num_articles) + '.txt'
save_file = 'even_rows'+ str(num_articles) + '.txt'
out_file_name_temp = './data/splitted_%d_'+ str(num_articles) + '.txt'
input_train_txt = './data/splitted_1_'+ str(num_articles) + '.txt'
input_valid_txt = './data/splitted_2_'+ str(num_articles) + '.txt'
processed_train_txt = './data/train_X_'+ str(num_articles) + '.txt'
processed_valid_txt = './data/valid_X_'+ str(num_articles) + '.txt'

c = 0
#titleを除いてファイルを作る
with open(filename,'w') as f:
    for directory in sorted(glob.glob('./extracted/*')):
        for name in sorted(glob.glob(directory+'/*')):
            with open(name, 'r') as r:
                for line in r:
                    if c >= num_articles:
                        break
                    # titleを削除する
                    if '<doc ' in line:
                        next(r)
                        next(r)
                    elif '</doc>' in line:
                        f.write('\n')
                        c += 1
                        continue
                    else:
                        # 空白・改行削除、大文字を小文字に変換
                        #text = line.strip('\n').lower()
                        text = line.lower()
                        text = text.replace('.\n', '. ')
                        text = text.replace('."', '". ')
                        text = text.replace('.\'\'', '\'\'. ')
                        text = text.strip('\n')
                        f.write(text)

#連続する文章を用意するため2n個の文章になるよう調整, 段落の区切りを無視
with open(save_file, 'w') as f:
    with open(filename) as r:
        for text in r:
            # 一文ごとに分割する
            text = text.split('. ')
            text = [t.strip() for t in text if t]
            # 一単元の文書が偶数個の文章から成るようにする(BERTのデータセットの都合上)
            max_text_len = len(text) // 2
            text = text[:max_text_len * 2]
            text = '\n'.join(text)
            f.write(text)

num_lines = sum(1 for line in open(save_file))
#print('Base file lines : ', num_lines)
train_lines = int(num_lines * p)
#print('Train file lines : ', train_lines)


split_index = 1
line_index = 1
out_file = open(out_file_name_temp % (split_index,), 'w')
in_file = open(save_file)
line = in_file.readline()

#trainとvalidの分離
while line:
    if line_index > train_lines:
        #print('Starting file: %d' % split_index)
        out_file.close()
        split_index = split_index + 1
        line_index = 1
        out_file = open(out_file_name_temp % (split_index,), 'w')
    out_file.write(line)
    line_index = line_index + 1
    line = in_file.readline()
    
out_file.close()
in_file.close()


print('Train file lines : ', sum(1 for line in open(input_train_txt)))
print('Valid file lines : ', sum(1 for line in open(input_valid_txt)))

# 偶数行の文章を奇数行の文章と接続するメソッド
def load_data(path):
    with open(path, encoding='utf-8') as f:
        even_rows = []
        odd_rows = []
        all_f = f.readlines()
        for row in all_f[::2]:
            even_rows.append(row.strip().replace('\n', ''))
        for row in all_f[1::2]:
            odd_rows.append(row.strip().replace('\n', ''))
    min_rows_len = int(min(len(even_rows), len(odd_rows)))
    even_rows = even_rows[:min_rows_len]
    odd_rows = odd_rows[:min_rows_len]

    concat_rows = []
    for even_r, odd_r in zip(even_rows, odd_rows):
        concat_r = '\t'.join([even_r, odd_r])
        concat_rows.append(concat_r)
    return concat_rows



train_data = load_data(input_train_txt)
valid_data = load_data(input_valid_txt)

# ランダムに並び替える
random.shuffle(train_data)
random.shuffle(valid_data)

with open(processed_train_txt, 'w') as f:
    f.write('\n'.join(train_data))
with open(processed_valid_txt, 'w') as f:
    f.write('\n'.join(valid_data))