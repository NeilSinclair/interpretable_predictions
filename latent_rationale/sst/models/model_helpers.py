#!/usr/bin/env python

from latent_rationale.sst.models.baseline import Baseline
from latent_rationale.sst.models.rl import RLModel
from latent_rationale.sst.models.latent import LatentRationaleModel


def build_model(model, tokenizer, cfg):

    vocab_size = tokenizer.vocab_size
    output_size = vocab_size

    # emb_size = cfg["embed_size"]
    emb_size = 50265 # Hard coded based on BART params
    # hidden_size = cfg["hidden_size"]
    hidden_size = 768 # Hard coded based on BART params
    dropout = cfg["dropout"]
    layer = cfg["layer"]
    dependent_z = cfg.get("dependent_z", False)

    selection = cfg["selection"]
    lasso = cfg["lasso"]

    sparsity = cfg["sparsity"]
    coherence = cfg["coherence"]

    assert 0 < selection <= 1.0, "selection must be in (0, 1]"

    lambda_init = cfg["lambda_init"]
    lagrange_lr = cfg["lagrange_lr"]
    lagrange_alpha = cfg["lagrange_alpha"]
    return LatentRationaleModel(
        vocab_size=vocab_size, emb_size=emb_size,
        hidden_size=hidden_size, output_size=output_size,
        vocab=None, dropout=dropout, layer=layer,
        dependent_z=dependent_z,
        selection=selection, lasso=lasso,
        lambda_init=lambda_init,
        lagrange_lr=lagrange_lr, lagrange_alpha=lagrange_alpha,
        model=model, tokenizer=tokenizer)
