#
# basic public information for transformer model
#

name = "bloomz-7b1-mt"
tokenizer = "tokenizer.json"

vocab_size = 250680
layers = 30
heads = 16
hiddens = 4096
head_hiddens = hiddens // heads

def _input():
    total = 0;
    ## vocab embedd
    total += vocab_size * hiddens;
    ## first layernorm form word embeddings
    total += 2 * hiddens;
    return total;
input_parameters = _input();

def _one_layer():
    total = 0;
    ## input layernorm
    total += 2 * hiddens;
    ## kqv
    total += 3 * hiddens * hiddens + 3 * hiddens;
    ## dense
    total += hiddens * hiddens + hiddens;
    ## post layernorm
    total += 2 * hiddens;
    ## mlp
    total += 4 * hiddens * hiddens + 4 * hiddens;
    total += 4 * hiddens * hiddens + hiddens;
    return total
one_layer_parameters = _one_layer();

def _output():
    total = 0;
    ## head layernorm
    total += 2 * hiddens;
    ## lm_head
    total += vocab_size * hiddens;
    return total;
output_parameters = _output();

def one_layer_workspace(max_batch, max_len):
    total = 0;
    # xinput & grad
    total += 2 * max_batch * max_len * hiddens;
    # layernorm's var, mean
    total += 3 * hiddens;
    # xa, xb, xc
    total += 3 * max_batch * max_len * hiddens;
    # kqv
    total += 3 * max_batch * max_len * hiddens;
    # xatten_in & xatten_out
    total += 2 * max_batch * max_len * hiddens;
    # x4a , x4b
    total += 4 * max_batch * max_len * hiddens + 4 * max_batch * max_len * hiddens
    # xll
    total += max_batch * heads * max_len * max_len;
    return total;

one_layer_forward_code =
'''
    ;; layernorm for input
    "xinput" @ "ln_mean" @ "input_ln_var" @ "input_layernorm.weight" @ "input_layernorm.bias" @ "xattn_in" @  "LN_EPS" @  op.layernorm

    ;; get kqv & transpose
    {
        "xattn_in" @ "query.weight" @ "query.bias" @ "xa" @ op.linear
        "xattn_in" @ "key.weight" @ "key.bias" @ "xb" @ op.linear
        "xattn_in" @ "value.weight" @ "value.bias" @ "xc" @ op.linear

        "ya" @ "zquery" @ op.transpos_0213
        "yb" @ "zkey" @ op.transpos_0213
        "yc" @ "zvalue" @ op.transpos_0213
    }

    ;; attention
    {
        ;; get query@key
        "zquery" @  "zkey" @  "xll" @ op.querykey

        ;; added alibi and apply xmask
        "xll" @ "alibi" @ "xll" @ op.add
        "xll" @ "xmask" @ "xll" @ op.add

        ;; do softmax
        "xll" @ "xll" @ op.softmax

        ;; do attention and transpose back
        "xll" @ "zvalue" @  "zc" @ op.attn          ;; attn -> zc
        "zc" @ "yattn_out" @ op.transpos_0213
    }

    ;; do dense & residual
    "xattn_out" @ "dense.weight" @ "dense.bias" @  "xb" @ op.linear
    "xb" @ "xinput" @ "xa" @ op.add

    ;; post layernorm
    "xa" @ "ln_mean" @ "post_ln_var" @ "post_attention_layernorm.weight" @ "post_attention_layernorm.bias" @ "xb" @  "LN_EPS" @  op.layernorm

    ;; MLP
    {
        ;; xa atteion output
        ;; xb passed post layernorm
        ;; 4h dense & glue

        "xb" @ "dense_h_to_4h.weight" @ "dense_h_to_4h.bias" @ "x4a" @ op.linear
        "x4a" @ "x4b" @ op.gelu

        ;; 4h dense
        "x4b" @ "dense_4h_to_h.weight" @ "dense_4h_to_h.bias" @ "xc" @ op.linear

        ;; residual
        "xa" @ "xc" @ "xa" @ op.add
    }
''';
