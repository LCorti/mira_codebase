import sys

from functools import partial
from torch.nn import Dropout
from transformers.models.t5 import modeling_t5
from transformers.models.t5.modeling_t5 import (
    T5DenseActDense,
    T5DenseGatedActDense,
    T5LayerNorm,
)

from app.vendor.lxt.efficient.patches import patch_method
from app.vendor.lxt.efficient.patches import (
    layer_norm_forward,
    mlp_forward,
    gated_mlp_forward,
    dropout_forward,
    replace_module,
)

attnLRP = {
    T5DenseActDense: partial(patch_method, mlp_forward),
    T5DenseGatedActDense: partial(patch_method, gated_mlp_forward),
    T5LayerNorm: partial(patch_method, layer_norm_forward),
    Dropout: partial(patch_method, dropout_forward),
    modeling_t5: partial(replace_module, sys.modules[__name__]),
}
