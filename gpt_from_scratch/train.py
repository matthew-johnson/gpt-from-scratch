import requests
import torch
import torch.nn as nn
from torch.nn import functional as F

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

#retrieve and encode training dataset
def load_dataset(url):
    try:
        response = requests.get(url) 
    except requests.exceptions.RequestException as e:
        print("Error occurred while fetching the dataset:", e)
    text = response.content.decode('utf-8')
    #read it
    with open('input.txt', 'w', encoding='utf-8') as f:
        f.write(text)

    #all unique charts that appear in the text
    tED = TextEncoderDecoder(text) 
    vocab_size = len(tED.chars)

    #encode the entire text dataset using our encoding function
    data = torch.tensor(tED.encode(text), dtype=torch.long)

    #split the data into training and validation sets
    n = int(len(data)*0.9)
    train_data = data[:n]
    val_data = data[n:]
    data = {'train':train_data, 'val':val_data}

    return data, vocab_size, tED

#get the batches from input data to be used to train the model
def get_batch(split, input_data, device, batch_size = 128, block_size = 256, seed=1337):
    torch.manual_seed(seed)
    data = input_data[split]
    if split == 'train':
        data = input_data['train']
    else:
        data = input_data['val']
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

@torch.no_grad()
def estimate_loss(input_data, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split,input_data,device)
            logits, loss = model(X, targets=Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size, block_size, n_embed, num_heads, num_layers, device='cuda', dropout=0.3):
        super().__init__()
        #each token reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        #adding embedding of positions as an additional input
        self.positional_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.ModuleList([Block(n_embed=n_embed, num_heads=num_heads, dropout=dropout, block_size=block_size) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.block_size = block_size
        self.device = device
        self.apply(self._init_weights)
        print(f"Model initialized with vocab_size={vocab_size}, block_size={block_size}, n_embed={n_embed}, num_heads={num_heads}, num_layers={num_layers}, dropout={dropout}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        """
        x: main input sequence (B, T)
        context: additional input sequence (B, T_context)
        targets: target sequence (B, T)
        """
        B, T = x.shape
        pos_emb = self.positional_embedding_table(torch.arange(T, device=self.device)) #T,C
        tok_emb = self.token_embedding_table(x) #B, T, C
        x_emb = tok_emb + pos_emb #B, T, C
        
        for block in self.blocks:
            x_emb = block(x_emb) #B, T, C

        x_emb = self.ln_f(x_emb) #B, T, C
        logits = self.lm_head(x_emb) #B, T, vocab_size
        
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
            #crop input x to the last block_size tokens
            x_cond = x[:, -self.block_size:]
            
            #get the predictions
            logits, loss = self.forward(x_cond)
            
            #focus only on the last time step
            logits = logits[:, -1, :] #becomes (batch_size, vocab_size) (B,C)
            
            #apply softmax to get the probabilities
            probs = F.softmax(logits, dim=-1) # (batch_size, vocab_size) (B,C)
            
            #sample 1 from the probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # (batch_size, 1) (B,1)
            
            #append sampled index to the running sequence
            x = torch.cat((x, idx_next), dim=1)
        return x

class FeedFoward(nn.Module):

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    #a single block of the transformer
    def __init__(self, num_heads, block_size, n_embed, dropout):
        super().__init__()
        head_size = n_embed // num_heads
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = MultiHeadAttention(head_size=head_size, num_heads=num_heads, block_size=block_size, n_embed=n_embed, dropout=dropout)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ffwd = FeedFoward(n_embed, dropout)
    
    def forward(self, x):
        #self attention
        x = x + self.attn(self.ln1(x))
        #feed forward
        x = x + self.ffwd(self.ln2(x))
        return x

#Attention is a communication mechanism between vector spaces that can improve model performance. 
#It allows each head to attend to a subset of the input tokens, which helps in capturing different aspects of the input data.
#This is a scaled dot product self-attention example ("self" because value, key, and query are all produced by the same input)
class MultiHeadAttention(nn.Module):
    #multiple heads of self-attention in parallel
    def __init__(self, head_size, block_size, n_embed, num_heads, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, block_size, n_embed, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)
        #dropout is a regularization technique used to prevent overfitting. It randomly sets some of the weights to zero during training.
        #this is meant to improve generalization of the model

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        #concatenate the outputs of all heads along the last dimension (C) to form the final output
        out = self.dropout(self.proj(out))
        #apply dropout to the output of the projection layer
        return out
        
class Head(nn.Module):
    def __init__(self, head_size, block_size, n_embed, dropout):
        super().__init__()
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
        self.head_size = head_size
        self.block_size = block_size
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T,C = x.shape
        # Ensure T does not exceed block_size
        T = min(T, self.block_size)
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)
        #compute attention scores "affinities"
        wei = q @ k.transpose(-2,-1) * (self.head_size ** -0.5) # (B,T,C) @ (B,T,C) -> (B,T,T)
        #normalizing by dividing by sqrt(head_size) in order to avoid softmax instability in scaling
        
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B,T,T) 
        wei = F.softmax(wei, dim=-1) # (B,T,T) 
        
        #apply attention weights to values, weighted aggregation
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B,T,C) @ (B,T,C) -> (B,T,C)
        return out

torch.manual_seed(1337)
data, vocab_size, encoder_decoder = load_dataset("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_iters = 5000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4
num_heads = 6
num_layers = 6

model = GPTLanguageModel(vocab_size, block_size=256, n_embed=256, device=device, num_heads=num_heads, num_layers=num_layers)
model = model.to(device)

#create pytorch Adam opitimizer and assign learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters -1:
        losses = estimate_loss(data, device)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train', data, device)
    
    # evaluate the loss
    logits, loss = model(xb, targets=yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate output for trained model
idx = torch.zeros((1, 1), dtype=torch.long, device=device) # (B,C) (1,1) assigned zeros
print(encoder_decoder.decode(model.generate(idx, max_new_tokens=500)[0].tolist()))
