import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32 # runs 4 batches at the same time
block_size = 8 # takes in 8 characters at the same time
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
# ------

torch.manual_seed(1337) 

with open('input.txt', 'r',encoding='utf-8') as f:
    text = f.read()

# these are all the unique characters that occur in this text
chars = sorted(list(set(text))) # sorts the values of the chars into a new set list
vocab_size = len(chars) # 65 total characters

# create mapping from characters to integers
stoi = {ch:i for i, ch in enumerate(chars)} # using the chars dictionary it creates a number for each char and calls it to make it a integer
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s:[stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Split data into train and validation sets
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # Train 90 percent of the values and let the rest be the validation
train_data = data[:n]
val_data = data[n:]

# Gets a batch of the value of the string
def get_batch(split):
    # Splits the values into train data and val data
    data = train_data if split == 'train' else val_data
    # makes a number for the length of ix
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # makes a stack with the data with the block size using the random number to get the value necessary
    x = torch.stack([data[i:i+block_size] for i in ix])
    # makes a stack of the values one greater than the ones in x
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y
    
# We will not be calling .backwars on this 
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # Call train again
    model.train()
    return out


# Bigram language model is a subclass of neural net module in pytorch 
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # makes a embeddding table of size vocab by vocab to get the logits for the next token using a lookup table
        # 24 in the xb table will go to the embedding table and look at the row of values in the embedding table row 24 
        # 
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Second embedding 
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        # idx is the current context of the character in a batch
        # Batch = 4
        # Time =8 
        # Channel = vocab_size
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        # x holds the character but also the position they occur in 
        x = tok_emb + pos_emb
        logits = self.lm_head(x)
        # the logits are the scores for the next character in the sequence 
        # Basically trying to find what comes next from the identity of a single token 

        if targets is None:
            loss = None
        else :
             # Reshape logits for pytorch library
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # A good way to see how good the scores where at predicting we use the loss fucntion
            # how well are we predicting the next character based on the logits
            loss = F.cross_entropy(logits, targets)
            
        # Expecting the loss to be -ln(1/65)
        return logits, loss

    # Getting the indices to pass into the next probability but now this is just looking at the past value
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Get predictions with the current indices
            logits, loss = self(idx)
            # Focus only on the last step pluc out the last element in the time dimension 
            logits = logits[:, -1, :] 
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # use multinomial distribution to sample from the probabilities and get a sample 
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # what is generated in idx_next is concatonated ontop of the previous idx to create the (B,T + 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
        

model = BigramLanguageModel()
model.to(device)

# Optimizing the learning 
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# Training loop for the loss statements
batch_size = 32
for iter in range(max_iters):
    # Calls the loss every one in a while to make sure its not to wrong
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample batch of data
    xb, yb = get_batch('train')
    # Evaluate the loss 
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate the model
print(decode(model.generate(idx = torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=500)[0].tolist()))

