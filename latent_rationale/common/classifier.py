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
                 nonlinearity: str = "sigmoid"
                 ):

        super(Classifier, self).__init__()

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

    def forward(self, final):

        # predict sentiment from final state(s)
        y = self.output_layer(final)

        return y
