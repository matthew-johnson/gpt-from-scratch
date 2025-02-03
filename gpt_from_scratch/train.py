import requests
import torch

def sort_text(text):
    return sorted(list(set(text)))
 
class TextEncoderDecoder:
    def __init__(self, text):
        self.chars = sort_text(text)
        self.char_to_int = {char: i for i, char in enumerate(self.chars)}
        self.int_to_char = {i: char for i, char in enumerate(self.chars)}

    def decode(self, encoded_text):
        return ''.join(self.int_to_char[i] for i in encoded_text)

    def encode(self, text):
        return [self.char_to_int[c] for c in text]

#get training dataset
def load_dataset(url):
    try:
        response = requests.get(url) 
    except requests.exceptions.RequestException as e:
        print("Error occurred while fetching the dataset:", e)
    text = response.content.decode('utf-8')
    #read it
    with open('input.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    #print("length of dataset in characters: ", len(text))

    #all unique charts that appear in the text
    e_d = TextEncoderDecoder(text) 
    vocab_size = len(e_d.chars)
    #print(''.join(chars))
    #print(vocab_size)

    #test encode and decode
    #print(e_d.encode("hello world"))
    #print(e_d.decode(e_d.encode("hello world")))

    #encode the entire text dataset using our encoding function
    data = torch.tensor(e_d.encode(text), dtype=torch.long)
    #print(data.shape, data.dtype)
    #print(data[:1000]) #example of how the characters will look to the model

    #split the data into training and validation sets
    n = int(len(data)*0.9)
    train_data = data[:n]
    val_data = data[n:]
    data = {'train':train_data, 'val':val_data}
    #print(train_data.shape, val_data.shape)
    #print(data)
    return data, vocab_size, e_d

#get the batches from input data to be used to train the model
def get_batch(split, input_data, batch_size = 4, block_size = 8, seed=1337):
    torch.manual_seed(seed)
    if split == 'train':
        data = input_data['train']
    else:
        data = input_data['val']
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        #each token reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, targets=None):
        # x and targets are both of shape: (batch_size, time_steps)
        logits = self.token_embedding_table(x)
        if targets is None:
            loss = None
        else:
            #reshape the logits and targets to be (batch_size * time_steps, vocab_size) or (B*T, C) for use in pytorch.nn.functional.cross_entropy
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, x, max_new_tokens):
        #x is of shape (batch_size, time_steps)
        for _ in range(max_new_tokens):
            #get the predictions
            logits, loss = self.forward(x)
            #focus only on the last time step
            logits = logits[:, -1, :] #becomes (batch_size, vocab_size) (B,C)
            #apply softmax to get the probabilities
            probs = F.softmax(logits, dim=-1) # (batch_size, vocab_size) (B,C)
            #sample 1 from the probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # (batch_size, 1) (B,1)
            #append sampled index to the running sequence
            x = torch.cat((x, idx_next), dim=1)
        return x

data, vocab_size, encoder_decoder = load_dataset("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
xb, yb = get_batch('train', data)
torch.manual_seed(1337)
m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
#print(logits.shape, logits.dtype)
print(loss)
idx = torch.zeros((1, 1), dtype=torch.long) # (B,C) (1,1) assigned zeros
#generate output for random model
print(encoder_decoder.decode(m.generate(idx, max_new_tokens=100)[0].tolist()))

#create pytorch Adam opitimizer and assign learning rate
optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)

#training loop x iterations
for steps in range(10000):
    xb, yb = get_batch('train', data, batch_size=32)
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())

#generate output for trained model
print(encoder_decoder.decode(m.generate(idx, max_new_tokens=500)[0].tolist()))

#self attention
B,T,C = 4,8,2 # batch size, sequence length, feature dimension
x = torch.randn(B,T,C) # (B,T,C) random tensor
x.shape

#weighted sum version of self attention
wei = torch.tril(torch.ones((T,T)))
wei = wei / wei.sum(dim=-1, keepdim=True)
xbow2 = wei @ x

#weighted sum version of self attention with softmax
tril = torch.tril(torch.ones((T,T)))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril==0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x