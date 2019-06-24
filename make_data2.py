from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import numpy as np
import random
import pickle
import tqdm
from collections import Counter


class TorchVocab(object):
    """
    :property freqs: collections.Counter, コーパス中の単語の出現頻度を保持するオブジェクト
    :property stoi: collections.defaultdict, string → id の対応を示す辞書
    :property itos: collections.defaultdict, id → string の対応を示す辞書
    """
    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>', '<oov>'],
                 vectors=None, unk_init=None, vectors_cache=None):
        """
        :param coutenr: collections.Counter, データ中に含まれる単語の頻度を計測するためのcounter
        :param max_size: int, vocabularyの最大のサイズ. Noneの場合は最大値なし. defaultはNone
        :param min_freq: int, vocabulary中の単語の最低出現頻度. この数以下の出現回数の単語はvocabularyに加えられない.
        :param specials: list of str, vocabularyにあらかじめ登録するtoken
        :param vecors: list of vectors, 事前学習済みのベクトル. ex)Vocab.load_vectors
        """
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list(specials)
        # special tokensの出現頻度はvocabulary作成の際にカウントされない
        for tok in specials:
            del counter[tok]

        max_size = None if max_size is None else max_size + len(self.itos)

        # まず頻度でソートし、次に文字順で並び替える
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
        
        # 出現頻度がmin_freq未満のものはvocabに加えない
        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        # dictのk,vをいれかえてstoiを作成する
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1


class Vocab(TorchVocab):
    def __init__(self, counter, max_size=None, min_freq=1):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        super().__init__(counter, specials=["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"], max_size=max_size, min_freq=min_freq)

    # override用
    def to_seq(self, sentece, seq_len, with_eos=False, with_sos=False) -> list:
        pass

    # override用
    def from_seq(self, seq, join=False, with_pad=False):
        pass

    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


# テキストファイルからvocabを作成する
class WordVocab(Vocab):
    def __init__(self, texts, max_size=None, min_freq=1, istitle=False):
        if istitle:
            print("Building Title_Vocab")
        else:
            print("Building Vocab")

        counter = Counter()
        for line in texts:
            #titleならば改行で区切ってcounterへ
            if istitle:
                words = line.split("\n")
            #titleじゃないならwordに分割してcounterへ
            else:
                if isinstance(line, list):
                    words = line
                else:
                    words = line.replace("\n", " ")
                    words = words.replace("\t", " ")
                    words = words.split()

            for word in words:
                counter[word] += 1
        super().__init__(counter, max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        if isinstance(sentence, str):
            sentence = sentence.split()

        seq = [self.stoi.get(word, self.unk_index) for word in sentence]

        if with_eos:
            seq += [self.eos_index]  # this would be index 1
        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        words = [self.itos[idx]
                 if idx < len(self.itos)
                 else "<%d>" % idx
                 for idx in seq
                 if not with_pad or idx != self.pad_index]

        return " ".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)


def build(corpus_path, output_path, vocab_size=None, encoding='utf-8', min_freq=1, istitle=False):
    with open(corpus_path, "r", encoding=encoding) as f:
        vocab = WordVocab(f, max_size=vocab_size, min_freq=min_freq, istitle=istitle)
    
    if istitle:
        print("TITLE_VOCAB SIZE:", len(vocab))
    else:
        print("VOCAB SIZE", len(vocab))

    vocab.save_vocab(output_path)



class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, is_train=True):
        self.vocab = vocab
        self.seq_len = seq_len
        self.is_train = is_train

        with open(corpus_path, "r", encoding=encoding) as f:
            self.datas = [line[:-1].split("\t") for line in f]#-1は\nを除くため

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        t1, (t2, is_next_label) = self.datas[item][0], self.random_sent(item)
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        t2 = t2_random + [self.vocab.eos_index]

        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        t2_label = t2_label + [self.vocab.pad_index]

        #文のペアの頭からseq_lenまででmasked language modelをしている
        #seq_len以降のwordsはmasked language modelに利用されていない!!
        segment_label = [1 for _ in range(len(t1))] + [2 for _ in range(len(t2))]
        bert_input = t1 + t2
        bert_label = t1_label + t2_label
        if len(bert_input) > self.seq_len:
            bert_input = bert_input[:self.seq_len]
            bert_label = bert_label[:self.seq_len]
        #segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        #bert_input = (t1 + t2)[:self.seq_len]#多分これだと最後の1つが削られるかも
        #bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []

        for i, token in enumerate(tokens):
            if self.is_train: # Trainingの時は確率的にMASKする
                prob = random.random()
            else:  # Predictionの時はMASKをしない
                prob = 1.0
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)

        return tokens, output_label

    def random_sent(self, index):
        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return self.datas[index][1], 1
        else:
            return self.datas[random.randrange(len(self.datas))][1], 0


class BERTftDataset(Dataset):
    def __init__(self, corpus_path, vocab, title_path, title_list, seq_len, encoding="utf-8", corpus_lines=None, is_train=True, shift=False, p=[0.15, 0.8, 0.1]):
        self.vocab = vocab
        self.title_list = title_list
        self.seq_len = seq_len
        self.is_train = is_train

        self.datas = []
        with open(corpus_path, "r", encoding=encoding) as f:
            for line in f:
                self.datas.append(line[:-1].replace("\t", " <eos> "))#-1は\nを除くため
        with open(title_path, "r", encoding=encoding) as f:
            self.titles = [line[:-1] for line in f]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item, shift=False):
        title = self.title_list.index(self.titles[item])
        t_random = self.random_word(self.datas[item])
        bert_input = [self.vocab.sos_index] + t_random + [self.vocab.eos_index]
        segment_label = [1 for _ in range(len(bert_input))]

        if len(bert_input) > self.seq_len:
            if shift:
                s = np.random.randint(len(bert_input)+1 - self.seq_len)
                bert_input = bert_input[0] + bert_input[1+s:s+self.seq_len]
            else:
                bert_input = bert_input[:self.seq_len]
            segment_label = segment_label[:self.seq_len]
        else:
            padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
            bert_input.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "segment_label": segment_label,
                  "title": title}        

        return {key: torch.tensor(value) for key, value in output.items()}


    def random_word(self, sentence, p=[0.15, 0.8, 0.1]):
        #p[0] = ratio of change
        #p[1] = ratio of mask in change
        #p[2] = ratio of random in change
        tokens = sentence.split()
        #output_label = []

        for i, token in enumerate(tokens):
            if self.is_train: # Trainingの時は確率的にMASKする
                prob = random.random()
            else:  # Predictionの時はMASKをしない
                prob = 1.0
            if prob < p[0]:
                prob /= p[0]

                # 80% randomly change token to mask token
                if prob < p[1]:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < p[1]+p[2]:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                #output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                #output_label.append(0)

        return tokens#, output_label
