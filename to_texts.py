import glob
import datetime
import os

num_words = []
titles = []
max_len = 50
for directory in sorted(glob.glob('./extracted/*')):#sortedすると10000のときデータが約10倍に
    d = directory[-2:]
    for name in sorted(glob.glob(directory+'/*')):
        with open(name, 'r') as r:
            for line in r:
                if '<doc ' in line:
                    next(r)
                    next(r)
                    title = line[line.rfind("title=")+7:-3]
                    title = title.replace(' ', '_').replace('/', '_')
                    if len(title) > max_len:
                        title = title[:max_len]
                    lines = ' '
                elif '</doc>' in line:
                    with open('./articles/'+d+'/'+title+'.txt', 'w') as f:
                        f.write(lines[1:])
                    num_words.append(len(lines.split()))
                    titles.append(title)
                else:
                    line = line.replace('\t', ' ').replace('\n', ' ')
                    lines += line
    print(directory, datetime.datetime.now())
with open('title_num_words', 'w') as f:
    for i in range(len(titles)):
        f.write(titles[i]+'\t'+str(num_words[i])+'\n')