from torch import nn
import numpy as np

from latent_rationale.common.latent import shift_tokens_right


class Classifier(nn.Module):
    """
    The Encoder takes an input text (and rationale z) and computes p(y|x,z)

    Supports a sigmoid on the final result (for regression)
    If not sigmoid, will assume cross-entropy loss (for classification)

    """

    def __init__(self,
                 embed:        nn.Embedding = None,
                 hidden_size:  int = 768,
                 output_size:  int = 1,
                 dropout:      float = 0.1,
                 layer:        str = "rcnn",
                 nonlinearity: str = "sigmoid",
                 model = None
                 ):

        super(Classifier, self).__init__()

        emb_size = embed.weight.shape[1]

        # The "encoding layer" is actually just the whole model here as we use the output of the
        # decoder layer
        self.enc_layer = model

        self.embed_layer = model.get_input_embeddings()

        enc_size = hidden_size
        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(enc_size, output_size),
            nn.Sigmoid() if nonlinearity == "sigmoid" else nn.LogSoftmax(dim=-1)
        )

        self.report_params()

    def report_params(self):
        # This has 1604 fewer params compared to the original, since only 1
        # aspect is trained, not all. The original code has 5 output classes,
        # instead of 1, and then only supervise 1 output class.
        count = 0
        for name, p in self.named_parameters():
            if p.requires_grad and "embed" not in name:
                count += np.prod(list(p.shape))
        print("{} #params: {}".format(self.__class__.__name__, count))

    def forward(self, x, mask, z=None):

        rnn_mask = mask
        encoder_emb = self.embed_layer(x)
        decoder_emb = self.embed_layer(shift_tokens_right(x))
        # apply z to main inputs
        if z is not None:
            z_mask = (mask.float() * z).unsqueeze(-1)  # [B, T, 1]
            rnn_mask = z_mask.squeeze(-1) > 0.  # z could be continuous
            encoder_emb = encoder_emb * z_mask
            decoder_emb = decoder_emb * z_mask

        # z is also used to control when the encoder layer is active
        lengths = mask.long().sum(1)

        # encode the sentence
        outputs = self.dec_layer(input_ids=None, attention_mask=mask,
                                 inputs_embeds=encoder_emb,
                                 decoder_inputs_embeds=decoder_emb)

        # Get the first token of the hidden state, the <CLS> token
        final = outputs.last_hidden_state[:, 1, :]

        # predict sentiment from final state(s)
        y = self.output_layer(final)

        return y
