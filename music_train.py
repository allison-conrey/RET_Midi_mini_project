import torch
from torch import nn
import numpy as np
import data_proc as dp
import results as res
import model_configuration as conf
import sys


gpu_available = torch.cuda.is_available()
device = torch.device('cpu')
if gpu_available:
    device = torch.device('cuda')


def split_into_batches(data, batch_size, seq_length):
    """
    Create a generator that returns batches of size
    """

    total_batch_size = batch_size * seq_length
    # total number of batches we can make
    batch_count = len(data) // total_batch_size

    # Keep only enough characters to make full batches
    data = data[:batch_count * total_batch_size]
    # Reshape into batch_size rows
    data = data.reshape((batch_size, -1))

    # iterate through the array, one sequence at a time
    for n in range(0, data.shape[1], seq_length):
        # The features
        x = data[:, n:n + seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], data[:, n + seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], data[:, 0]
        yield x, y


class musicTransformer(nn.Module):
    def __init__(self, vocab, n_head=4, head_dim=16, n_layers=1, lr=0.001, dropout_prob=0.4):
        super().__init__()
        self.d_model = n_head * head_dim
        self.n_layers = n_layers
        # self.n_hidden = n_hidden  # nhead
        self.dropout_prob = dropout_prob
        self.lr = lr

        self.vocab = vocab
        self.int2char = dict(enumerate(self.vocab))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        """ nn.TransformerDecoderLayer
        d_model (int) – the number of expected features in the input (required).
        nhead (int) – the number of heads in the multiheadattention models (required).
        dim_feedforward (int) – the dimension of the feedforward network model (default=2048).
        dropout (float) – the dropout value (default=0.1).
        activation (Union[str, Callable[[Tensor], Tensor]]):
            the activation function of the intermediate layer, can be a string (“relu” or “gelu”) or a unary callable. Default: relu
        layer_norm_eps (float) – the eps value in layer normalization components
            (default=1e-5).
        batch_first (bool):
            If True, then the input and output tensors are provided as (batch, seq, feature).
            Default: False (seq, batch, feature).
        norm_first (bool): if True, layer norm is done prior to self attention,
            multihead attention and feedforward operations, respectively.
            Otherwise it’s done after.
            Default: False (after).
        bias (bool) – If set to False, Linear and LayerNorm layers will not learn an additive bias. Default: True.
        """
        # TODO: use Karpathy miniGPT architecture - https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
        # self.transformer = nn.TransformerDecoderLayer(
        #     d_model=len(self.vocab),
        #     nhead=4)

        """ nn.Transformer()
        d_model (int) – the number of expected features in the encoder/decoder inputs (default=512).
        nhead (int) – the number of heads in the multiheadattention models (default=8).
        num_encoder_layers (int) – the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers (int) – the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward (int) – the dimension of the feedforward network model (default=2048).
        dropout (float) – the dropout value (default=0.1).
        activation (Union[str, Callable[[Tensor], Tensor]]):
            the activation function of encoder/decoder intermediate layer, can be a string (“relu” or “gelu”) or a unary callable.
            Default: relu
        custom_encoder (Optional[Any]):
            custom encoder (default=None).
        custom_decoder (Optional[Any]): custom decoder (default=None).
        layer_norm_eps (float): the eps value in layer normalization components (default=1e-5).
        batch_first (bool): True=then the input and output tensors are provided as (batch, seq, feature).
            Default: False (seq, batch, feature).
        norm_first (bool):
            True: encoder and decoder layers will perform LayerNorms before other attention and feedforward operations, otherwise after.
            Default: False (after).
        bias (bool): If set to False, Linear and LayerNorm layers will not learn an additive bias. Default: True.
        """
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.n_head,
            num_encoder_layers=self.n_layers,
            num_decoder_layers=self.n_layers
        )
        """ LSTM layer
        number of tokens in vocab = number of features of input
        n_hidden = number of lstm units per layer
        n_layers = number of hidden layers
        """
        # self.lstm = nn.LSTM(len(self.vocab), n_hidden, n_layers,
        #                     dropout_prob=dropout_prob, batch_first=True)
        # dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        """
        output layer
        applies y = xA ^ T + b
        """
        self.fc = nn.Linear(self.n_hidden, len(self.vocab))

    def forward(self, x, hidden_init):
        transformer_out, hidden_state = self.transformer(x, hidden_init)
        dropout_out = self.dropout(transformer_out)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        dropout_out = dropout_out.contiguous().view(-1, self.n_hidden)

        # last layer
        out = self.fc(dropout_out)

        # return final output and hidden_state
        return out, hidden_state

    def init_hidden(self, batch_size):
        """LSTM's have two separate states(hidden and memory)
        denoted as state_h and state_c respectively which is why to sets of weights are initialised
        """
        weights = next(self.parameters()).data
        if (gpu_available):
            hidden = (
                weights.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(device),
                weights.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(device)
            )
        else:
            hidden = (weights.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weights.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden


class musicRNN(nn.Module):
    def __init__(self, vocab, n_hidden=512, n_layers=3, lr=0.001, dropout_prob=0.4):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.dropout_prob = dropout_prob
        self.lr = lr

        self.tokens = vocab
        self.int2char = dict(enumerate(self.tokens))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        """
        LSTM layer
        number of tokens = number of features of input
        n_hidden = number of lstm units per layer
        n_layers = number of hidden layers
        """
        self.lstm = nn.LSTM(
            len(self.tokens), n_hidden, n_layers,
            dropout_prob=dropout_prob, batch_first=True
        )

        # dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        """
        output layer
        applies y = xA ^ T + b
        """
        self.fc = nn.Linear(n_hidden, len(self.tokens))

    def forward(self, x, hidden_init):
        lstm_out, hidden_state = self.lstm(x, hidden_init)

        # pass to dropout layer
        out = self.dropout(lstm_out)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.n_hidden)

        # last layer
        out = self.fc(out)

        # return final output and hidden_state
        return out, hidden_state

    def init_hidden(self, batch_size):
        """LSTM's have two separate states(hidden and memory)
        denoted as state_h and state_c respectively which is why to sets of weights are initialised
        """
        weights = next(self.parameters()).data
        if (gpu_available):
            hidden = (
                weights.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(device),
                weights.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(device)
            )
        else:
            hidden = (weights.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weights.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden


def split_train_valid(data, val_frac):
    """
    splits data into training and validatiioin
    """
    valid_idx = int(len(data) * (1 - val_frac))
    return data[:valid_idx], data[valid_idx:]

# TRAINING


def train(model, data, vocab, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # training and validation data
    data, valid_data = split_train_valid(data, val_frac)

    if (gpu_available):
        model.cuda()

    count = 0
    vocab_count = len(vocab)
    loss_arr = []
    val_loss_arr = []
    epochs_to_plot = []

    log = ""

    for e in range(epochs):
        # hidden state init
        h = model.init_hidden(batch_size)

        for x, y in split_into_batches(data, batch_size, seq_length):
            count += 1

            x = dp.one_hot_from_vocab(x, vocab_count)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if (gpu_available):
                inputs, targets = inputs.cuda(), targets.cuda()

            h = tuple([each.data for each in h])

            model.zero_grad()

            output, h = model(inputs, h)

            loss = criterion(output, targets.view(batch_size * seq_length).long())
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            # loss stats
            if count % print_every == 0:
                # Get validation loss
                val_h = model.init_hidden(batch_size)
                val_losses = []
                model.eval()
                for x, y in split_into_batches(valid_data, batch_size, seq_length):
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    x = dp.one_hot_from_vocab(x, vocab_count)

                    val_h = tuple([each.data for each in val_h])

                    inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
                    if(gpu_available):
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = model(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size * seq_length).long())

                    val_losses.append(val_loss.item())

                model.train()  # reset to train mode after iterationg through validation data

                loss_arr.append(loss.item())
                val_loss_arr.append(np.mean(val_losses))
                epochs_to_plot.append(e + 1)

                log_info = "Epoch: {}/{}... Step: {}... Loss: {:.4f}... Val Loss: {:.4f}\n".format(
                    e + 1, epochs, count, loss.item(), np.mean(val_losses))
                log += log_info
                print(log_info)
    res.save_log(log, "model_loss_log.txt")
    res.save_plot_results(epochs_to_plot, loss_arr, val_loss_arr,
                          "Training and Validation Loss for: {} Layers,  \n{} Hidden Units, and a {} Learning Rate".format(
                              model.n_layers, model.n_hidden, model.lr),
                          "model_loss_plot.png")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if (len(argv) > 1):
        print("Usage: python3 music_train.py [model_name.pth]")
        sys.exit()

    if (len(argv) == 1):
        model_name = argv[0]
    else:
        model_name = 'model.pth'

    if (gpu_available):
        print("GPU available. Training on GPU!")
    else:
        print("GPU unavailable. Training on CPU")

    data = dp.get_track_data(conf.path, False)
    vocab = dp.get_vocab(data)
    int2char = dict(enumerate(vocab))
    char2int = {ch: ii for ii, ch in int2char.items()}

    int_encoded = np.array([char2int[ch] for ch in data])

    # model = musicTransformer(vocab, conf.n_hidden, conf.n_layers, conf.lr, conf.dropout)

    model = musicTransformer(vocab, n_hidden=conf.n_hidden, n_layers=1, lr=conf.lr, dropout_prob=conf.dropout)
    print(model)
    train(model, int_encoded, vocab, epochs=conf.n_epochs, batch_size=conf.batch_size,
          seq_length=conf.seq_length, clip=conf.grad_clip, lr=conf.lr,
          val_frac=conf.val_frac, print_every=conf.print_every)
    checkpoint = {'n_hidden': model.n_hidden,
                  'n_layers': model.n_layers,
                  'state_dict': model.state_dict(),
                  'tokens': model.tokens,
                  'int2char': model.int2char,
                  'char2int': model.char2int}

    with open(model_name, 'wb') as f:
        torch.save(checkpoint, f)
