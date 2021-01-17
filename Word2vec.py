import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

from collections import Counter
import numpy as np
import random
import scipy

USE_CUDA = torch.cuda.is_available()
device = torch.device('cpu')

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if USE_CUDA:
    torch.cuda.manual_seed(1)
    device = torch.device('cuda:0')

# 超参数
C = 3  # context window
K = 100  # number of negative samples
NUM_EPOCHS = 2
MAX_VOCAB_SIZE = 30000  # 单词表数目，其他词使用UNK表示
BATCH_SIZE = 128
LEARNING_RATE = 0.2
EMBEDDING_SIZE = 100

text = ''
with open('./text8/text8.train.txt', 'r') as fin:
    text = fin.read()
text = text.split()
vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))
vocab['<unk>'] = len(text) - np.sum(list(vocab.values()))
idx_to_word = [word for word in vocab.keys()]
word_to_idx = {word: i for i, word in enumerate(idx_to_word)}


def word_tokenize(text):
    return text.split()


def find_nearest(word, embedding_weights):
    index = word_to_idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [idx_to_word[i] for i in cos_dis.argsort()[:10]]


class WordEmbeddingDataSet(tud.Dataset):
    def __init__(self, text, word_to_idx, idx_to_word, word_freqs, word_counts):
        super(WordEmbeddingDataSet, self).__init__()
        self.text_encoded = [word_to_idx.get(word, word_to_idx['<unk>']) for word in text]
        self.text_encoded = torch.LongTensor(self.text_encoded)
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)

    # 数据集item数量
    def __len__(self):
        return len(self.text_encoded)

    def __getitem__(self, idx):
        center_word = self.text_encoded[idx]
        # window内单词下标
        pos_indics = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))
        # 取余，防止超出text长度
        pos_indics = [i % len(self.text_encoded) for i in pos_indics]
        # 周围单词
        pos_words = self.text_encoded[pos_indics]
        # 负例采样单词
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)
        return center_word, pos_words, neg_words


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        # 初始化对下降速度影响很大
        initrange = 0.5 / self.embed_size
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.in_embed.weight.data.uniform_(-initrange, initrange)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.out_embed.weight.data.uniform_(-initrange, initrange)


    def forward(self, input_labels, pos_labels, neg_labels):
        # input_labels: [BATCH_SIZE]
        # pos_labels: [BATCH_SIZE, (window_size * 2)]
        # neg_labels: [BATCH_SIZE, (window_size * 2 * K)]
        input_embedding = self.in_embed(input_labels)  # [BATCH_SIZE, embed_size]
        pos_embedding = self.out_embed(pos_labels)  # [BATCH_SIZE, (window_size * 2), embed_size]
        neg_embedding = self.out_embed(neg_labels)  # [BATCH_SIZE, (window_size * 2 * K), embed_size]

        input_embedding = input_embedding.unsqueeze(2)  # [BATCH_SIZE, embed_size, 1]
        pos_dot = torch.bmm(pos_embedding, input_embedding).squeeze()  # [BATCH_SIZE, (window_size * 2)]
        neg_dot = torch.bmm(neg_embedding, -input_embedding).squeeze()  # [BATCH_SIZE, (window_size * 2 * K)]

        log_pos = F.logsigmoid(pos_dot).sum(1)
        log_neg = F.logsigmoid(neg_dot).sum(1)

        loss = log_pos + log_neg
        return -loss  # [BATCH_SIZE]

    def input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()


def main():
    word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
    word_freqs = word_counts / np.sum(word_counts)
    word_freqs = word_freqs ** (3. / 4.)
    word_freqs = word_freqs / np.sum(word_freqs)
    VOCAB_SIZE = len(idx_to_word)

    dataset = WordEmbeddingDataSet(text, word_to_idx, idx_to_word, word_freqs, word_counts)
    dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # for i, (center_word, pos_words, neg_words) in enumerate(dataloader):
    #     print(center_word, pos_words, neg_words)
    #     if i >= 1:
    #         break

    model = EmbeddingModel(VOCAB_SIZE, EMBEDDING_SIZE)
    if USE_CUDA:
        model = model.to(device)

    test_words = ["good", "fresh", "monster", "green", "like", "america", "chicago", "work", "computer", "language"]
    # training
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    for e in range(NUM_EPOCHS):
        for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
            if USE_CUDA:

                input_labels = input_labels.long().to(device)
                pos_labels = pos_labels.long().to(device)
                neg_labels = neg_labels.long().to(device)

            optimizer.zero_grad()
            loss = model(input_labels, pos_labels, neg_labels).mean()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("epoch", e, "iteration", i, loss.item())

            if i % 2000 == 0:
                embedding_weights = model.input_embeddings()
                for word in test_words:
                    print(word, find_nearest(word, embedding_weights))

    # test
    embedding_weights = model.input_embeddings()
    for word in test_words:
        print(word, find_nearest(word, embedding_weights))


if __name__ == '__main__':
    main()
