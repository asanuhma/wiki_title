import datetime
from torch.utils.data import DataLoader
from bert import BERT, BERTLM
from make_data import BERTDataset, build, WordVocab
from train0 import BERTTrainer
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

dt_now = str(datetime.datetime.now()).replace(' ', '')

num_articles = 10000
min_words = 5000
max_words = 10000
min_len = 100
dir_name = str(num_articles) + '_' + str(min_words) + '_' + str(max_words) + '_' + str(min_len)
#train_dataset = './data/_train_X_'+ str(num_articles) + '.txt'
#test_dataset = './data/_valid_X_'+ str(num_articles) + '.txt'
train_dataset = './data1/'+dir_name+'/train_X_'+dir_name
test_dataset = './data1/'+dir_name+'/valid_X_'+dir_name
vocab_path='./data1/'+dir_name+'/vocab'+ dt_now
output_model_path='./outputs1/'+dir_name+'/bertmodel'+ dt_now
load_path = None#"./output/bertmodel2019-05-0817:02:15.612442.ep30"

if not os.path.isdir('./results1/'+dir_name):
    os.makedirs('./results1/'+dir_name)
if not os.path.isdir('./outputs1/'+dir_name):
    os.makedirs('./outputs1/'+dir_name)
if not os.path.isdir('./fig1/'+dir_name):
    os.makedirs('./fig1/'+dir_name)

hidden=32 #768
layers=2 #12
attn_heads=2 #12
seq_len=60

batch_size=16
epochs=50
num_workers=5

log_freq=2000
corpus_lines=None
with_cuda=True
lr=1e-3
adam_weight_decay=0.00
adam_beta1=0.9
adam_beta2=0.999

dropout=0.1

min_freq=6

#corpus_path = './data/_train_X_'+ str(num_articles) + '.txt'
corpus_path = train_dataset
label_path=None

build(corpus_path, vocab_path, min_freq=min_freq)

print("Loading Vocab", vocab_path)
vocab = WordVocab.load_vocab(vocab_path)

print("Loading Train Dataset", train_dataset)
train_dataset = BERTDataset(train_dataset, vocab, seq_len=seq_len, label_path=label_path, corpus_lines=corpus_lines)

print("Loading Test Dataset", test_dataset)
test_dataset = BERTDataset(test_dataset, vocab, seq_len=seq_len, label_path=label_path) if test_dataset is not None else None

print("Creating Dataloader")
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers) if test_dataset is not None else None

if load_path == None:
    print("Building BERT model")
    bert = BERT(len(vocab), hidden=hidden, n_layers=layers, attn_heads=attn_heads, dropout=dropout)
else:
    print("Loading BERT model from " + load_path)
    bert = torch.load(load_path)


print("Creating BERT Trainer")
trainer = BERTTrainer(epochs, bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                      lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay,
                      with_cuda=with_cuda, log_freq=log_freq)

print("Training Start")
for epoch in range(epochs):
    trainer.train(epoch)
    # Model Save
    if (epoch+1) % 10 == 0:
        trainer.save(epoch, output_model_path)
    trainer.test(epoch)

n = np.arange(epochs)
train_acc = trainer.train_accs[::2]
test_acc = trainer.train_accs[1::2]
train_loss = trainer.train_lossses[::2]
test_loss = trainer.train_lossses[1::2]
results = [train_acc, test_acc, train_loss, test_loss]
#np.savetxt('./results/n'+str(num_articles)+'_h'+str(hidden)+'_l'+str(layers)+'_a'+str(attn_heads)+'_s'+str(seq_len)+'_b'+str(batch_size)+'_e'+str(epochs)+'_d'+str(dropout).replace('.', '')+'_m'+str(min_freq), results)
np.savetxt('./results1/'+dir_name+'/pt_'+dir_name+'_h'+str(hidden)+'_l'+str(layers)+'_a'+str(attn_heads)+'_s'+str(seq_len)+'_b'+str(batch_size)+'_e'+str(epochs)+'_d'+str(dropout).replace('.', '')+'_m'+str(min_freq), results)

A = plt.figure("acc")
plt.plot(n, train_acc)
plt.plot(n, test_acc)
#plt.savefig('./fig/a_n'+str(num_articles)+'_h'+str(hidden)+'_l'+str(layers)+'_a'+str(attn_heads)+'_s'+str(seq_len)+'_b'+str(batch_size)+'_e'+str(epochs)+'_d'+str(dropout).replace('.', '')+'_m'+str(min_freq))
plt.savefig('./fig1/'+dir_name+'/pta_'+dir_name+'_h'+str(hidden)+'_l'+str(layers)+'_a'+str(attn_heads)+'_s'+str(seq_len)+'_b'+str(batch_size)+'_e'+str(epochs)+'_d'+str(dropout).replace('.', '')+'_m'+str(min_freq))
L = plt.figure("loss")
plt.plot(n, train_loss)
plt.plot(n, test_loss)
#plt.savefig('./fig/l_n'+str(num_articles)+'_h'+str(hidden)+'_l'+str(layers)+'_a'+str(attn_heads)+'_s'+str(seq_len)+'_b'+str(batch_size)+'_e'+str(epochs)+'_d'+str(dropout).replace('.', '')+'_m'+str(min_freq))
plt.savefig('./fig1/'+dir_name+'/ptl_'+dir_name+'_h'+str(hidden)+'_l'+str(layers)+'_a'+str(attn_heads)+'_s'+str(seq_len)+'_b'+str(batch_size)+'_e'+str(epochs)+'_d'+str(dropout).replace('.', '')+'_m'+str(min_freq))
plt.show()