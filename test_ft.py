import datetime
from torch.utils.data import DataLoader
from bert import BERT, BERTLM
from make_data import build, WordVocab
from make_data2 import BERTDataset, BERTftDataset
from train1 import BERTTrainer
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import pickle

dt_now = str(datetime.datetime.now()).replace(' ', '')

num_articles = 10000
min_words = 5000
max_words = 10000
min_len = 100
#max_p = 0.5
max_m = 100
max_p = max_m/num_articles
k = [1, 10, 50]

dir_name = str(num_articles) + '_' + str(min_words) + '_' + str(max_words) + '_' + str(min_len)
train_dataset = './data1/'+dir_name+'/train_X_'+dir_name
test_dataset = './data1/'+dir_name+'/valid_X_'+dir_name
train_title = './data1/'+dir_name+'/train_T_'+dir_name# + '.txt'
test_title = './data1/'+dir_name+'/valid_T_'+dir_name# + '.txt'
ft_train_dataset = './data1/'+dir_name+'/ft_train_X_'+dir_name+'_'+str(max_p).replace('.', '')
ft_test_dataset = './data1/'+dir_name+'/ft_valid_X_'+dir_name+'_'+str(max_p).replace('.', '')
p_list_pkl = './data1/'+dir_name+'/p_list'+dir_name+'.pkl'

if not os.path.isdir('./results1/'+dir_name):
    os.makedirs('./results1/'+dir_name)
if not os.path.isdir('./outputs1/'+dir_name):
    os.makedirs('./outputs1/'+dir_name)
if not os.path.isdir('./fig1/'+dir_name):
    os.makedirs('./fig1/'+dir_name)

with open(p_list_pkl, 'rb') as f:
    p_list = pickle.load(f)

vocab_path='./data1/'+dir_name+'/vocab'+ dt_now
output_model_path='./outputs1/'+dir_name+'/bertmodel'+ dt_now
load_path = None#"./outputs1/"+dir_name+"/bertmodel2019-05-2513:43:20.349333.ep49"

hidden=32 #768
layers=2 #12
attn_heads=2 #12
seq_len=250

batch_size=8
epochs=20
num_workers=5

log_freq=200
corpus_lines=None
with_cuda=True
lr=1e-3
adam_weight_decay=0.00
adam_beta1=0.9
adam_beta2=0.999

dropout=0.0#0.1
min_freq=6
shift=False
mask_p = 0.15
learning_decay = True
sh = '_sh' if shift else ''
ld = '_ld' if learning_decay else ''

#corpus_path = './data/_train_X_'+ str(num_articles) + '.txt'
corpus_path = train_dataset
label_path=None

def for_ft(dataset, title, ft_dataset, p_list, max_p):
    with open(dataset, 'r') as f:
        data = [line[:-1] for line in f]
    with open(title, 'r') as f:
        title = [line[:-1] for line in f]
    
    ft_data = []
    num_w, num_m = 0, 0
    for d, t in zip(data, title):
        ft_d = [['<mask>' if p_list[t][w] > max_p else w for w in s.split()] for s in d.split('\t')]
        num_m += sum([ft_s.count('<mask>') for ft_s in ft_d])
        num_w += sum(len(ft_s) for ft_s in ft_d)
        ft_data.append('\t'.join([' '.join(ft_s) for ft_s in ft_d]))
    with open(ft_dataset, 'w') as f:
        for d in ft_data:
            f.write(d+'\n')
    return num_m/num_w

train_mask_ratio = for_ft(train_dataset, train_title, ft_train_dataset, p_list, max_p)
test_mask_ratio = for_ft(test_dataset, test_title, ft_test_dataset, p_list, max_p)
print("max_p:", max_p, "train_mask_ratio:", train_mask_ratio, "test_mask_ratio:", test_mask_ratio, "mask_p:", mask_p)

build(corpus_path, vocab_path, min_freq=min_freq)
print("Loading Vocab", vocab_path)
vocab = WordVocab.load_vocab(vocab_path)

#build(title_path, title_vocab_path, istitle=True)
#print("Loading Title_Vocab", title_vocab_path)
#title_vocab = WordVocab.load_vocab(title_vocab_path)
title_list = list(p_list)

print("Loading Train Dataset", ft_train_dataset)
train_dataset = BERTftDataset(ft_train_dataset, vocab, train_title, title_list, seq_len=seq_len, corpus_lines=corpus_lines, shift=shift, p=[max((mask_p-train_mask_ratio)/(1-train_mask_ratio), 0), 1, 0])

print("Loading Test Dataset", ft_test_dataset)
test_dataset = BERTftDataset(ft_test_dataset, vocab, test_title, title_list, seq_len=seq_len, shift=shift, p=[max((mask_p-test_mask_ratio)/(1-test_mask_ratio), 0), 1, 0]) if test_dataset is not None else None

print("Creating Dataloader")
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers) if test_dataset is not None else None

if load_path == None:
    pt = ''
    print("Building BERT model")
    bert = BERT(len(vocab), hidden=hidden, n_layers=layers, attn_heads=attn_heads, dropout=dropout)
else:
    pt = '_pt'
    print("Loading BERT model from " + load_path)
    bert = torch.load(load_path)


print("Creating BERT Trainer")
trainer = BERTTrainer(epochs, bert, len(vocab), len(title_list), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                      lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay,
                      with_cuda=with_cuda, log_freq=log_freq)

print("Training Start")
for epoch in range(epochs):
    trainer.trainf(epoch, learning_decay=learning_decay, k=k)
    # Model Save
    if (epoch+1) % 10 == 0:
        trainer.save(epoch, output_model_path)
    trainer.testf(epoch, k=k)

n = np.arange(epochs)
train_acc = np.array(trainer.train_accs[::2]).T
test_acc = np.array(trainer.train_accs[1::2]).T
train_loss = np.array(trainer.train_lossses[::2])
test_loss = np.array(trainer.train_lossses[1::2])
results = np.concatenate([train_acc, test_acc, train_loss.reshape(1, -1), test_loss.reshape(1, -1)])#, train_loss, test_loss])
np.savetxt('./results1/'+dir_name+'/ft_'+dir_name+'_'+str(max_m).replace('.', '')+str(test_mask_ratio).replace('.', '')+'_h'+str(hidden)+'_l'+str(layers)+'_a'+str(attn_heads)+'_s'+str(seq_len)+'_b'+str(batch_size)+'_e'+str(epochs)+'_d'+str(dropout).replace('.', '')+'_m'+str(min_freq)+'_mp'+str(mask_p).replace('.', '')+pt+sh+ld+'_'+str(k).replace(' ',''), results)

color = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
A = plt.figure("acc")
for i in range(len(k)):
    plt.plot(n, train_acc[i], color=color[i], linestyle="solid")
    plt.plot(n, test_acc[i], color=color[i], linestyle="dashed")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.savefig('./fig1/'+dir_name+'/fta_'+dir_name+'_'+str(max_m).replace('.', '')+str(test_mask_ratio).replace('.', '')+'_h'+str(hidden)+'_l'+str(layers)+'_a'+str(attn_heads)+'_s'+str(seq_len)+'_b'+str(batch_size)+'_e'+str(epochs)+'_d'+str(dropout).replace('.', '')+'_m'+str(min_freq)+'_mp'+str(mask_p).replace('.', '')+pt+sh+ld+'_'+str(k).replace(' ',''))
L = plt.figure("loss")
plt.plot(n, train_loss)
plt.plot(n, test_loss)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig('./fig1/'+dir_name+'/ftl_'+dir_name+'_'+str(max_m).replace('.', '')+str(test_mask_ratio).replace('.', '')+'_h'+str(hidden)+'_l'+str(layers)+'_a'+str(attn_heads)+'_s'+str(seq_len)+'_b'+str(batch_size)+'_e'+str(epochs)+'_d'+str(dropout).replace('.', '')+'_m'+str(min_freq)+'_mp'+str(mask_p).replace('.', '')+pt+sh+ld+'_'+str(k).replace(' ',''))
plt.show()