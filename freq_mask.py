import pickle
import numpy as np
import matplotlib.pyplot as plt

min_words = 5000
max_words = 10000
num_articles = 100
min_len = 100
max_p = 0.99

dir_name = str(num_articles) + '_' + str(min_words) + '_' + str(max_words) + '_' + str(min_len)

train_data_txt = './data1/'+dir_name+'/train_X_'+dir_name# + '.txt'
valid_data_txt = './data1/'+dir_name+'/valid_X_'+dir_name# + '.txt'
train_title_txt = './data1/'+dir_name+'/train_T_'+dir_name# + '.txt'
valid_title_txt = './data1/'+dir_name+'/valid_T_'+dir_name# + '.txt'
ft_train_data_txt = './data1/'+dir_name+'/ft_train_X_'+dir_name+'_'+str(max_p).replace('.', '')
ft_valid_data_txt = './data1/'+dir_name+'/ft_valid_X_'+dir_name+'_'+str(max_p).replace('.', '')
p_list_pkl = './data1/'+dir_name+'/p_list'+dir_name+'.pkl'

with open(p_list_pkl, 'rb') as f:
    p_list = pickle.load(f)

def for_ft(data_txt, title_txt, p_list, max_p):
    with open(data_txt, 'r') as f:
        data = [line[:-1] for line in f]
    with open(title_txt, 'r') as f:
        title = [line[:-1] for line in f]
    
    #ft_data = []
    num_w, num_m = 0, 0
    for d, t in zip(data, title):
        ft_d = [['<mask>' if p_list[t][w] > max_p else w for w in s.split()] for s in d.split('\t')]
        num_m += sum([ft_s.count('<mask>') for ft_s in ft_d])
        num_w += sum(len(ft_s) for ft_s in ft_d)
        #ft_data.append('\t'.join([' '.join(ft_s) for ft_s in ft_d]))
    #with open(ft_data_txt, 'w') as f:
    #    for d in ft_data:
    #        f.write(d+'\n')    
    #print('max_p:', max_p, 'mask_p:', num_m/num_w)
    return num_m/num_w

x = np.arange(21)*10/num_articles
#x = np.arange(11)/10
yt = np.zeros_like(x)
yv = np.zeros_like(x)
yy = np.zeros_like(x)
for i in range(len(x)):
    yt[i] = for_ft(train_data_txt, train_title_txt, p_list, x[i])
plt.plot(x*num_articles, yt)
#plt.plot(x, yt)

num_articles = 1000
dir_name = str(num_articles) + '_' + str(min_words) + '_' + str(max_words) + '_' + str(min_len)
train_data_txt = './data1/'+dir_name+'/train_X_'+dir_name# + '.txt'
train_title_txt = './data1/'+dir_name+'/train_T_'+dir_name# + '.txt'
p_list_pkl = './data1/'+dir_name+'/p_list'+dir_name+'.pkl'
with open(p_list_pkl, 'rb') as f:
    p_list = pickle.load(f)
x = np.arange(21)*10/num_articles
for i in range(len(x)):
    yv[i] = for_ft(train_data_txt, train_title_txt, p_list, x[i])
plt.plot(x*num_articles, yv)
#plt.plot(x, yv)

num_articles = 10000
dir_name = str(num_articles) + '_' + str(min_words) + '_' + str(max_words) + '_' + str(min_len)
train_data_txt = './data1/'+dir_name+'/train_X_'+dir_name# + '.txt'
train_title_txt = './data1/'+dir_name+'/train_T_'+dir_name# + '.txt'
p_list_pkl = './data1/'+dir_name+'/p_list'+dir_name+'.pkl'
with open(p_list_pkl, 'rb') as f:
    p_list = pickle.load(f)
x = np.arange(21)*10/num_articles
for i in range(len(x)):
    yy[i] = for_ft(train_data_txt, train_title_txt, p_list, x[i])
plt.plot(x*num_articles, yv)
#plt.plot(x, yy)

#plt.xlabel('num of the word/total num of the word')
plt.xlabel('num of the word/average num of the word')
plt.ylabel('mask ratio')
plt.savefig('mask(p)_3')
plt.show()

#for_ft(train_data_txt, train_title_txt, ft_train_data_txt, p_list, max_p)
#for_ft(valid_data_txt, valid_title_txt, ft_valid_data_txt, p_list, max_p)
'''
with open(ft_train_data_txt, 'w') as f:
    for d in ft_train_data:
        f.write(d+'\n')
with open(ft_valid_data_txt, 'w') as f:
    for d in ft_valid_data:
        f.write(d+'\n')
'''