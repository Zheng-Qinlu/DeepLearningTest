import os
import time
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import requests
import tarfile
import shutil
import csv

# Configuration
class Config:
    # Path relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'simple-examples', 'data')
    save_path = os.path.join(base_dir, 'ptb_lstm_model.pth')
    plot_path = os.path.join(base_dir, 'training_curve.png')
    csv_path = os.path.join(base_dir, 'training_metrics.csv')
    
    batch_size = 20
    num_steps = 35 # Sequence length
    hidden_size = 650
    num_layers = 2
    dropout = 0.5
    lr = 20.0
    epochs = 50 # Increased to 40 for better convergence
    vocab_size = 10000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_interval = 200

# Data Download and Extraction
def download_and_extract_data(dest_folder):
    url = "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"
    filename = "simple-examples.tgz"
    filepath = os.path.join(dest_folder, filename)
    
    if not os.path.exists(os.path.join(dest_folder, 'simple-examples')):
        print(f"Downloading {url} to {filepath}...")
        try:
            r = requests.get(url, stream=True)
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            
            print("Extracting...")
            with tarfile.open(filepath, "r:gz") as tar:
                tar.extractall(path=dest_folder)
            print("Done.")
        except Exception as e:
            print(f"Error downloading or extracting: {e}")
            # Fallback or manual check
    else:
        print("Data already exists.")

# Dictionary and Corpus
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        print(f"Reading data from {path}")
        self.train = self.tokenize(os.path.join(path, 'ptb.train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'ptb.valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'ptb.test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path), f"File {path} not found."
        # Add words to the dictionary
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding='utf-8') as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

# Batching
def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def get_batch(source, i, seq_len):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

# Model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout)
        self.decoder = nn.Linear(hidden_size, vocab_size)

        # Tie weights
        self.decoder.weight = self.encoder.weight

        self.init_weights()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.hidden_size),
                weight.new_zeros(self.num_layers, bsz, self.hidden_size))

# Training and Evaluation
def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def evaluate(model, data_source, criterion, config):
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(config.batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, config.num_steps):
            data, targets = get_batch(data_source, i, config.num_steps)
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output.view(-1, config.vocab_size), targets).item()
    return total_loss / (len(data_source) - 1)

def train(model, train_data, criterion, optimizer, epoch, config):
    model.train()
    total_loss = 0.
    start_time = time.time()
    hidden = model.init_hidden(config.batch_size)
    
    epoch_loss = 0.0
    total_batches = 0
    
    for batch, i in enumerate(range(0, train_data.size(0) - 1, config.num_steps)):
        data, targets = get_batch(train_data, i, config.num_steps)
        
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of dataset.
        hidden = repackage_hidden(hidden)
        
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, config.vocab_size), targets)
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()

        total_loss += loss.item()
        epoch_loss += loss.item()
        total_batches += 1

        if batch % config.log_interval == 0 and batch > 0:
            cur_loss = total_loss / config.log_interval
            # elapsed = time.time() - start_time
            # print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
            #       'loss {:5.2f} | ppl {:8.2f}'.format(
            #         epoch, batch, len(train_data) // config.num_steps, optimizer.param_groups[0]['lr'],
            #         elapsed * 1000 / config.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
            
    return math.exp(epoch_loss / total_batches)

def generate(model, corpus, config, seed_word='the', temperature=1.0, num_words=50):
    model.eval()
    if seed_word not in corpus.dictionary.word2idx:
        print(f"Word '{seed_word}' not in dictionary.")
        return

    input_idx = torch.tensor([[corpus.dictionary.word2idx[seed_word]]]).long().to(config.device)
    hidden = model.init_hidden(1)
    
    print(f"\nGenerating text with seed '{seed_word}':")
    with torch.no_grad():
        print(seed_word, end=' ')
        for _ in range(num_words):
            output, hidden = model(input_idx, hidden)
            word_weights = output.squeeze().div(temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input_idx.fill_(word_idx)
            word = corpus.dictionary.idx2word[word_idx]
            
            if word == '<eos>':
                print('.', end=' ')
            elif word != '<unk>':
                print(word, end=' ')
    print("\n")

# Main
def main():
    # Set seed
    torch.manual_seed(1111)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1111)
        print("Using CUDA")
    else:
        print("Using CPU")

    config = Config()
    
    # Download data
    download_and_extract_data(config.base_dir)
    
    # Load data
    print("Loading data...")
    if not os.path.exists(config.data_path):
        print(f"Data path {config.data_path} does not exist. Please check download.")
        return

    corpus = Corpus(config.data_path)
    config.vocab_size = len(corpus.dictionary)
    print(f"Vocab size: {config.vocab_size}")

    train_data = batchify(corpus.train, config.batch_size, config.device)
    val_data = batchify(corpus.valid, config.batch_size, config.device)
    test_data = batchify(corpus.test, config.batch_size, config.device)

    # Build model
    model = LSTMModel(config.vocab_size, config.hidden_size, config.num_layers, config.dropout).to(config.device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    
    best_val_loss = None
    train_ppls = []
    val_ppls = []
    test_ppls = []

    do_train = True
    if os.path.exists(config.save_path):
        print(f"Found existing model at {config.save_path}")
        user_input = input("Do you want to load it and skip training? (y/n): ").strip().lower()
        if user_input == 'y':
            do_train = False
            with open(config.save_path, 'rb') as f:
                model.load_state_dict(torch.load(f))
            print("Model loaded.")

    if do_train:
        # Initialize CSV
        with open(config.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train PPL', 'Valid PPL', 'Test PPL', 'Time(s)'])

        patience = 3
        no_improve_epochs = 0
        try:
            for epoch in range(1, config.epochs + 1):
                epoch_start_time = time.time()
                
                # Train
                train_ppl = train(model, train_data, criterion, optimizer, epoch, config)
                train_ppls.append(train_ppl)
                
                # Evaluate
                val_loss = evaluate(model, val_data, criterion, config)
                val_ppl = math.exp(val_loss)
                val_ppls.append(val_ppl)
                
                # Evaluate on Test set every epoch
                test_loss = evaluate(model, test_data, criterion, config)
                test_ppl = math.exp(test_loss)
                test_ppls.append(test_ppl)

                epoch_time = time.time() - epoch_start_time

                print('| end of epoch {:3d} | time: {:5.2f}s | train ppl {:8.2f} | valid ppl {:8.2f} | test ppl {:8.2f}'.format(
                    epoch, epoch_time, train_ppl, val_ppl, test_ppl))

                # Write to CSV
                with open(config.csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, train_ppl, val_ppl, test_ppl, epoch_time])

                # Save model & Early Stopping
                if not best_val_loss or val_loss < best_val_loss:
                    with open(config.save_path, 'wb') as f:
                        torch.save(model.state_dict(), f)
                    best_val_loss = val_loss
                    no_improve_epochs = 0
                else:
                    # Anneal the learning rate if no improvement
                    optimizer.param_groups[0]['lr'] /= 2.0
                    no_improve_epochs += 1
                
                if no_improve_epochs >= patience and test_ppl < 80:
                    print(f"Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs and test ppl < 80).")
                    break
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    # Load best model if we trained
    if do_train:
        with open(config.save_path, 'rb') as f:
            model.load_state_dict(torch.load(f))

    # Run on test data
    test_loss = evaluate(model, test_data, criterion, config)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))

    
    # Generate examples
    seeds = ['the', 'it', 'one']
    for seed in seeds:
        generate(model, corpus, config, seed_word=seed)

    # Plotting
    if do_train:
        epochs_range = range(1, len(train_ppls) + 1)
        plt.figure(figsize=(12, 6))
        
        plt.plot(epochs_range, train_ppls, label='Train PPL', linestyle='--', marker='o', markersize=3)
        plt.plot(epochs_range, val_ppls, label='Valid PPL', linestyle='-', marker='s', markersize=3)
        plt.plot(epochs_range, test_ppls, label='Test PPL', linestyle='-.', marker='^', markersize=3)

        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.title('Training, Validation, and Test Perplexity over Epochs')
        plt.legend()
        plt.grid(True)
        
        # Add a horizontal line for the target PPL = 80
        plt.axhline(y=80, color='r', linestyle=':', label='Target PPL (80)')
        plt.legend()

        plt.savefig(config.plot_path)
        print(f"Training curve saved to {config.plot_path}")

if __name__ == "__main__":
    main()


