#
# basic public information for transformer model
#

name = "bloomz-7b1-mt"
tokenizer = "tokenizer.json"

vocab_size = 250880
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


