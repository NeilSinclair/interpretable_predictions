import os
import argparse
import re
from collections import namedtuple
import numpy as np
import torch
import random
import math

from torch.utils.data import Dataset

from latent_rationale.sst.constants import UNK_TOKEN, PAD_TOKEN
from latent_rationale.sst.plotting import plot_heatmap
from torch.nn.init import _calculate_fan_in_and_fan_out
from torch import nn


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def find_ckpt_in_directory(path):
    for f in os.listdir(os.path.join(path, "")):
        if f.startswith('model'):
            return os.path.join(path, f)
    print("Could not find ckpt in {}".format(path))


def filereader(path):
    """read SST lines"""
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            yield line.strip().replace("\\", "")


def tokens_from_treestring(s):
    """extract the tokens from a sentiment tree"""
    # return re.findall(r"\([0-9] ([^\(\)]+)\)", s)
    # This has been edited by Neil Sinclair to work with the Yelp Dataset
    # Remove some extra spaces
    s = re.sub(r' {2,}', ' ', s)
    # return re.findall(r"([A-Za-z0-9_.?!',]*) *", s)[:-1]
    return s

def token_labels_from_treestring(s):
    """extract token labels from sentiment tree"""
    return list(map(int, re.findall(r"\(([0-9]) [^\(\)]", s)))


Example = namedtuple("Example", ["tokens", "label"])


def sst_reader(paths, labels = None, lower=False):
    """
    Reads in examples
    :param path:
    :param lower:
    :return:
    """
    for i, path in enumerate(paths):
        for line in filereader(path):
            line = line.lower() if lower else line
            line = re.sub("\\\\", "", line)  # fix escape
            tokens = tokens_from_treestring(line)
            if labels is not None:
                label = labels[i]
            else:
                label = int(line[1])
            # token_labels = token_labels_from_treestring(line)
            # assert len(tokens) == len(token_labels), "mismatch tokens/labels"
            yield Example(tokens=tokens, label=label)


def print_parameters(model):
    """Prints model parameters"""
    total = 0
    for name, p in model.named_parameters():
        total += np.prod(p.shape)
        print("{:24s} {:12s} requires_grad={}".format(name, str(list(p.shape)),
                                                      p.requires_grad))
    print("\nTotal parameters: {}\n".format(total))



def get_minibatch(data, batch_size=25, shuffle=False):
    """Return minibatches, optional shuffling"""

    if shuffle:
        print("Shuffling training data")
        random.shuffle(data)  # shuffle training data each epoch

    batch = []

    # yield minibatches
    for example in data:
        batch.append(example)

        if len(batch) == batch_size:
            yield batch
            batch = []

    # in case there is something left
    if len(batch) > 0:
        yield batch


def pad(tokens, length, pad_value=1):
    """add padding 1s to a sequence to that it has the desired length"""
    return tokens + [pad_value] * (length - len(tokens))

class CreateTokens(object):
  ''' Create the embeddings for the sentences passed into the model '''
  def __init__(self, tokenizer):
    ''' tokenizer - the BART tokenizer
        create_emb - the create_embedding() method from the BART model '''

    self.tokenizer = tokenizer

  def __call__(self, sample, label):
    ''' sample is a line from the text with columns [Original, Masked, Label] '''
    sample = encode_single_sentence(self.tokenizer, sample, label)

    return sample

class TokensDataSet(Dataset):
    ''' class for loading and transforming data into embeddings '''

    def __init__(self, data_file, transform=None, translate=False, labels=[0,1]):
        '''
        Args: data_file - the path to the data_dile
              transform - an instantiated transformation class
              translate - whether to encode a translation (i.e. swap the label)
        '''
        self.data = list(sst_reader(data_file, labels))
        self.transform = transform
        self.translate = translate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Get a data item and transform it
        sample = self.data[idx].tokens
        label = self.data[idx].label

        # If we're translating, swap the label from 0 to 1 or 1 to 0
        if self.translate:
            sample[-1] = 1 - sample[-1]

        if self.transform:
            sample = self.transform(sample, label)
        return sample

    ''' Code taken almost verbatim from utils.py in the transformers/seq2seq github '''

    def collate_fn(self, batch):
        input_ids = torch.stack([x["input_ids"] for x in batch]).squeeze()
        labels = torch.stack([x["labels"] for x in batch]).squeeze()

        batch = {
            "input_ids": input_ids,
            "labels": labels
        }
        return batch


def encode_single_sentence(tokenizer, source_sentence, label, max_length=16,
                           pad_to_max_length=True, return_tensors="pt",
                           add_special_tokens=True, device='cuda'):
    ''' Function that tokenizes a sentence
        Args: tokenizer - the BART tokenizer
              model_emb - the BART method create_embeddings()
              source_sentence - the <masked> source sentence: string
              target_sentence - the <masked> target sentence: string
              label - the label of the sentence: int
              max_length - max truncated/padded length
              return_targets - whether to return the tokenized targets; not necessary if we're just looking at the validation sentences
        Returns: Dictionary with keys: input_ids, attention_mask, target_ids
    '''

    input_ids = []
    attention_masks = []
    labels = []
    tokenized_sentences = {}
    # Remove unecessary tokens
    source_sentence = re.sub(r' {2,10}', ' ', source_sentence)
    encoded_dict = tokenizer(
        source_sentence,
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        truncation=True,
        return_tensors=return_tensors,
        add_prefix_space=False,
        add_special_tokens=add_special_tokens
    )

    input_ids.append(encoded_dict['input_ids'])
    # labels.append(label)

    input_ids = torch.cat(input_ids, dim=0).to(device)
    labels = torch.Tensor([label]).to(device)
    # attention_masks = torch.cat(attention_masks, dim=0)

    processed_sentence = {
        "input_ids": input_ids,
        "labels": labels
    }

    return processed_sentence


def prepare_minibatch(mb, vocab, tokenizer, device=None, sort=True, max_length=32):
    """
    Minibatch is a list of examples.
    This function converts words to IDs and returns
    torch tensors to be used as input/targets.
    """
    # batch_size = len(mb)
    reverse_map = None
    lengths = np.array([len(ex.tokens) for ex in mb])
    maxlen = max_length

    # vocab returns 0 if the word is not there
    # x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen) for ex in mb]

    x = tokenizer(
        mb,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors='pt',
        add_prefix_space=True,
        add_special_tokens = True
    )
    y = [ex.label for ex in mb]


    x = np.array(x['input_ids'])
    y = np.array(y)

    if sort:  # required for LSTM
        sort_idx = np.argsort(lengths)[::-1]
        x = x[sort_idx]
        y = y[sort_idx]

        # to put back into the original order
        reverse_map = np.zeros(len(lengths), dtype=np.int32)
        for i, j in enumerate(sort_idx):
            reverse_map[j] = i

    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)

    return x, y, reverse_map


def plot_dataset(model, data, batch_size=100, device=None, save_path=".",
                 ext="pdf"):
    """Accuracy of a model on given data set (using minibatches)"""

    model.eval()  # disable dropout
    sent_id = 0

    for mb in get_minibatch(data, batch_size=batch_size, shuffle=False):
        x, targets, reverse_map = prepare_minibatch(
            mb, model.vocab, device=device, sort=True)

        with torch.no_grad():
            logits = model(x)

            alphas = model.alphas if hasattr(model, "alphas") else None
            z = model.z if hasattr(model, "z") else None

        # reverse sort
        alphas = alphas[reverse_map] if alphas is not None else None
        z = z.squeeze(1).squeeze(-1)  # make [B, T]
        z = z[reverse_map] if z is not None else None

        for i, ex in enumerate(mb):
            tokens = ex.tokens

            if alphas is not None:
                alpha = alphas[i][:len(tokens)]
                alpha = alpha[None, :]
                path = os.path.join(
                    save_path, "plot{:04d}.alphas.{}".format(sent_id, ext))
                plot_heatmap(alpha, column_labels=tokens, output_path=path)

            # print(tokens)
            # print(" ".join(["%4.2f" % x for x in alpha]))

            # z is [batch_size, num_samples, time]
            if z is not None:

                zi = z[i, :len(tokens)]
                zi = zi[None, :]
                path = os.path.join(
                    save_path, "plot{:04d}.z.{}".format(sent_id, ext))
                plot_heatmap(zi, column_labels=tokens, output_path=path)

            sent_id += 1


def xavier_uniform_n_(w, gain=1., n=4):
    """
    Xavier initializer for parameters that combine multiple matrices in one
    parameter for efficiency. This is e.g. used for GRU and LSTM parameters,
    where e.g. all gates are computed at the same time by 1 big matrix.
    :param w:
    :param gain:
    :param n:
    :return:
    """
    with torch.no_grad():
        fan_in, fan_out = _calculate_fan_in_and_fan_out(w)
        assert fan_out % n == 0, "fan_out should be divisible by n"
        fan_out = fan_out // n
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        nn.init.uniform_(w, -a, a)


def initialize_model_(model):
    """
    Model initialization.

    :param model:
    :return:
    """
    # Custom initialization
    print("Glorot init")
    for name, p in model.named_parameters():
        if name.startswith("embed") or "lagrange" in name:
            print("{:10s} {:20s} {}".format("unchanged", name, p.shape))
        elif "lstm" in name and len(p.shape) > 1:
            print("{:10s} {:20s} {}".format("xavier_n", name, p.shape))
            xavier_uniform_n_(p)
        elif len(p.shape) > 1:
            print("{:10s} {:20s} {}".format("xavier", name, p.shape))
            torch.nn.init.xavier_uniform_(p)
        elif "bias" in name:
            print("{:10s} {:20s} {}".format("zeros", name, p.shape))
            torch.nn.init.constant_(p, 0.)
        else:
            print("{:10s} {:20s} {}".format("unchanged", name, p.shape))


def get_predict_args():
    parser = argparse.ArgumentParser(description='SST prediction')
    parser.add_argument('--ckpt', type=str, default="path_to_checkpoint",
                        required=True)
    parser.add_argument('--plot', action="store_true", default=False)
    args = parser.parse_args()
    return args


def get_args():
    parser = argparse.ArgumentParser(description='SST')
    parser.add_argument('--save_path', type=str, default='sst_results/default')
    parser.add_argument('--resume_snapshot', type=str, default='')

    parser.add_argument('--num_iterations', type=int, default=-25)
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--eval_batch_size', type=int, default=25)
    parser.add_argument('--embed_size', type=int, default=300)
    parser.add_argument('--proj_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=150)

    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--threshold', type=float, default=1e-4)
    parser.add_argument('--cooldown', type=int, default=5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--max_grad_norm', type=float, default=5.)

    parser.add_argument('--model',
                        choices=["baseline", "rl", "attention",
                                 "latent"],
                        default="baseline")
    parser.add_argument('--dist', choices=["", "hardkuma"],
                        default="")

    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=-1)
    parser.add_argument('--save_every', type=int, default=-1)

    parser.add_argument('--dropout', type=float, default=0.5)

    parser.add_argument('--layer', choices=["lstm"], default="lstm")
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')

    parser.add_argument('--dependent-z', action='store_true',
                        help="make dependent decisions for z")

    # rationale settings for RL model
    parser.add_argument('--sparsity', type=float, default=0.0)
    parser.add_argument('--coherence', type=float, default=0.0)

    # rationale settings for HardKuma model
    parser.add_argument('--selection', type=float, default=1.,
                        help="Target text selection rate for Lagrange.")
    parser.add_argument('--lasso', type=float, default=0.0)

    # lagrange settings
    parser.add_argument('--lagrange_lr', type=float, default=0.01,
                        help="learning rate for lagrange")
    parser.add_argument('--lagrange_alpha', type=float, default=0.99,
                        help="alpha for computing the running average")
    parser.add_argument('--lambda_init', type=float, default=1e-4,
                        help="initial value for lambda")

    # misc
    parser.add_argument('--word_vectors', type=str,
                        default='data/sst/glove.840B.300d.sst.txt')
    args = parser.parse_args()
    return args
