import torch
import torch.nn as nn
import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from bert import BERT, BERTLM

class BERTTrainer:
    def __init__(self, epochs, bert: BERT, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01,
                 with_cuda: bool = True, log_freq: int = 10):
        """
        :param bert: BERT model
        :param vocab_size: vocabに含まれるトータルの単語数
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: 学習率
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logを表示するiterationの頻度
        """

        # GPU環境において、GPUを指定しているかのフラグ
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        self.bert = bert
        self.model = BERTLM(bert, vocab_size).to(self.device)

        if torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)

        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.epochs = epochs
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        #self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

        # masked_token予測のためのLoss関数を設定
        self.criterion = nn.NLLLoss()
        self.log_freq = log_freq
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        
        self.train_lossses = []
        self.train_accs = []

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        :param epoch: 現在のepoch
        :param data_loader: torch.utils.data.DataLoader
        :param train: trainかtestかのbool値
        """
        str_code = "train" if train else "test"
        self.optim = Adam(self.model.parameters(), lr=self.lr*(1-epoch/self.epochs), betas=self.betas, weight_decay=self.weight_decay)
        data_iter = tqdm.tqdm(enumerate(data_loader), desc="EP_%s:%d" % (str_code, epoch), total=len(data_loader), bar_format="{l_bar}{r_bar}")


        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        if train:
            self.model.train()
        else:
            self.model.eval()
            
        for i, data in data_iter:
            # 0. batch_dataはGPU or CPUに載せる
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            next_sent_output, mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"])

            # 2-1. NLLLoss(negative log likelihood) : next_sentence_predictionのLoss
            next_loss = self.criterion(next_sent_output, data["is_next"])

            # 2-2. NLLLoss(negative log likelihood) : predicting masked token word
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])

            # 2-3. next_lossとmask_lossの合計をlossとする
            loss = next_loss + mask_loss

            # 3. training時のみ,backwardとoptimizer更新を行う
            if train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            # next sentence prediction accuracy
            correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data["is_next"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=", total_correct * 100.0 / total_element)
        self.train_lossses.append(avg_loss / len(data_iter))
        self.train_accs.append(total_correct * 100.0 / total_element)
        
    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path