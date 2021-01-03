import os
import time
from collections import OrderedDict
import json

import torch
import torch.optim
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler

from latent_rationale.common.util import make_kv_string
from latent_rationale.sst.vocabulary import Vocabulary
from latent_rationale.sst.models.model_helpers import build_model
from latent_rationale.sst.util import get_args, sst_reader, \
    prepare_minibatch, get_minibatch, print_parameters, \
    initialize_model_, get_device, TokensDataSet, encode_single_sentence, CreateTokens
from latent_rationale.sst.evaluate import evaluate

from transformers import BartTokenizer, BartModel, BartConfig, DistilBertModel, DistilBertTokenizer

device = get_device()
print("device:", device)


def freeze_pos_embeds(model, model_type="bart"):
    ''' freeze the positional embedding parameters of the model; adapted from finetune.py '''
    if model_type == "bart":
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_positions)
    else:
        freeze_params(model.embeddings.position_embeddings)


def freeze_token_embeds(model, model_type="bart"):
    ''' freeze the positional embedding parameters of the model; adapted from finetune.py '''
    if model_type == "bart":
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)
    else:
        freeze_params(model.embeddings.word_embeddings)

def freeze_params(model):
  ''' Function that takes a model as input (or part of a model) and freezes the layers for faster training
      adapted from finetune.py '''
  for layer in model.parameters():
    layer.requires_grade = False

def train():
    """
    Main training loop.
    """

    ## Begin by instantiating the BART model and tokenizer
    # tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', add_prefix_space=True)
    # bart_model = BartModel.from_pretrained(
    #     "facebook/bart-base")

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', add_prefix_space=True)
    bart_model = DistilBertModel.from_pretrained(
        "distilbert-base-uncased")

    cfg = get_args()
    cfg = vars(cfg)
    # Lots of model information given when training
    # for k, v in cfg.items():
    #     print("{:20} : {:10}".format(k, v))

    num_iterations = cfg["num_iterations"]
    print_every = cfg["print_every"]
    eval_every = cfg["eval_every"]
    batch_size = cfg["batch_size"]
    eval_batch_size = cfg.get("eval_batch_size", batch_size)

    print("Loading data")

    # Load the tokens into custom dataset objects
    dataset = TokensDataSet(data_file=["data/sst/sentiment.train.0", "data/sst/sentiment.train.1"],
                            transform=CreateTokens(tokenizer),
                            labels=[0, 1])
    train_data = DataLoader(dataset, collate_fn=dataset.collate_fn, sampler=RandomSampler(dataset), batch_size=batch_size)

    dataset = TokensDataSet(data_file=["data/sst/sentiment.test.0", "data/sst/sentiment.test.1"],
                            transform=CreateTokens(tokenizer),
                            labels=[0, 1])
    test_data = DataLoader(dataset, collate_fn=dataset.collate_fn,  batch_size=batch_size)

    dataset = TokensDataSet(data_file=["data/sst/sentiment.dev.0", "data/sst/sentiment.dev.1"],
                            transform=CreateTokens(tokenizer),
                            labels=[0, 1])
    dev_data = DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=batch_size)

    print("train", len(train_data))
    print("dev", len(dev_data))
    print("test", len(test_data))

    iters_per_epoch = len(train_data) // cfg["batch_size"]

    if cfg["eval_every"] == -1:
        eval_every = iters_per_epoch
        print("Set eval_every to {}".format(iters_per_epoch))

    if cfg["num_iterations"] < 0:
        num_iterations = iters_per_epoch * -1 * cfg["num_iterations"]
        print("Set num_iterations to {}".format(num_iterations))

    example = next(iter(dev_data))
    print("First train example:", [tokenizer.decode(w) for w in example['input_ids'][0]])
    print("First train example tokens:", example['input_ids'][0])
    print("First train example label:", example['labels'][0])

    writer = SummaryWriter(log_dir=cfg["save_path"])  # TensorBoard

    vocab = Vocabulary()  # populated by load_glove
    glove_path = cfg["word_vectors"]
    # vectors = load_glove(glove_path, vocab)

    # Map the sentiment labels 0-4 to a more readable form (and the opposite)
    i2t = ["very negative", "negative", "neutral", "positive", "very positive"]
    t2i = OrderedDict({p: i for p, i in zip(i2t, range(len(i2t)))})

    # Set the embeddings and encoder weights for the BART model to be non-trainable as we
    # only want the output
    print("Freezing Model parameters")
    freeze_token_embeds(bart_model, model_type = "distilbert")
    freeze_pos_embeds(bart_model, model_type = "distilbert")

    # Build model
    print("Building the model")
    model = build_model(bart_model, tokenizer, cfg, model_type = "bert")
    # initialize_model_(model)

    optimizer = Adam(model.parameters(), lr=cfg["lr"],
                     weight_decay=cfg["weight_decay"])

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg["lr_decay"], patience=cfg["patience"],
        verbose=True, cooldown=cfg["cooldown"], threshold=cfg["threshold"],
        min_lr=cfg["min_lr"])

    iter_i = 0
    train_loss = 0.
    print_num = 0
    start = time.time()
    losses = []
    accuracies = []
    best_eval = 1.0e9
    best_iter = 0

    model = model.to(device)

    # print model
    # print(model)
    # print_parameters(model)

    while True:  # when we run out of examples, shuffle and continue
        for i, batch in enumerate(train_data):
        # for batch in get_minibatch(train_data, batch_size=batch_size, shuffle=True):
            epoch = iter_i // iters_per_epoch

            model.train()
            x = batch['input_ids']
            targets = batch['labels']
            # x, targets, _ = prepare_minibatch(batch, model.vocab, tokenizer, device=device)

            mask = (x != tokenizer.pad_token_id)

            logits = model(x)  # forward pass

            loss, loss_optional = model.get_loss(logits, targets, mask=mask)
            model.zero_grad()  # erase previous gradients

            train_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=cfg["max_grad_norm"])
            optimizer.step()

            print_num += 1
            iter_i += 1

            # print info
            if iter_i % print_every == 0:

                train_loss = train_loss / print_every
                writer.add_scalar('train/loss', train_loss, iter_i)
                for k, v in loss_optional.items():
                    writer.add_scalar('train/'+k, v, iter_i)

                print_str = make_kv_string(loss_optional)
                min_elapsed = (time.time() - start) // 60
                print("Epoch %r Iter %r time=%dm loss=%.4f %s" %
                      (epoch, iter_i, min_elapsed, train_loss, print_str))
                losses.append(train_loss)
                print_num = 0
                train_loss = 0.

            # evaluate
            if iter_i % eval_every == 0:
                dev_eval = evaluate(model, dev_data,
                                    batch_size=eval_batch_size, device=device)
                accuracies.append(dev_eval["acc"])
                for k, v in dev_eval.items():
                    writer.add_scalar('dev/'+k, v, iter_i)

                print("# epoch %r iter %r: dev %s" % (
                    epoch, iter_i, make_kv_string(dev_eval)))

                # save best model parameters
                compare_score = dev_eval["loss"]
                if "obj" in dev_eval:
                    compare_score = dev_eval["obj"]

                scheduler.step(compare_score)  # adjust learning rate

                if (compare_score < (best_eval * (1-cfg["threshold"]))) and \
                        iter_i > (3 * iters_per_epoch):
                    print("***highscore*** %.4f" % compare_score)
                    best_eval = compare_score
                    best_iter = iter_i

                    for k, v in dev_eval.items():
                        writer.add_scalar('best/dev/' + k, v, iter_i)

                    if not os.path.exists(cfg["save_path"]):
                        os.makedirs(cfg["save_path"])

                    ckpt = {
                        "state_dict": model.state_dict(),
                        "cfg": cfg,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_eval": best_eval,
                        "best_iter": best_iter
                    }
                    path = os.path.join(cfg["save_path"], "model.pt")
                    torch.save(ckpt, path)

            # done training
            if iter_i == num_iterations:
                print("# Done training")

                # evaluate on test with best model
                print("# Loading best model")
                path = os.path.join(cfg["save_path"], "model.pt")
                if os.path.exists(path):
                    ckpt = torch.load(path)
                    model.load_state_dict(ckpt["state_dict"])
                else:
                    print("No model found.")

                print("# Evaluating")
                dev_eval = evaluate(
                    model, dev_data, batch_size=eval_batch_size,
                    device=device)
                test_eval = evaluate(
                    model, test_data, batch_size=eval_batch_size,
                    device=device)

                print("best model iter {:d}: "
                      "dev {} test {}".format(
                        best_iter,
                        make_kv_string(dev_eval),
                        make_kv_string(test_eval)))

                # save result
                result_path = os.path.join(cfg["save_path"], "results.json")

                cfg["best_iter"] = best_iter

                for k, v in dev_eval.items():
                    cfg["dev_" + k] = v
                    writer.add_scalar('best/dev/' + k, v, iter_i)

                for k, v in test_eval.items():
                    print("test", k, v)
                    cfg["test_" + k] = v
                    writer.add_scalar('best/test/' + k, v, iter_i)

                writer.close()

                with open(result_path, mode="w") as f:
                    json.dump(cfg, f)

                return losses, accuracies


if __name__ == "__main__":
    train()
