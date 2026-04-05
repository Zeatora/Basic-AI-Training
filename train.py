import torch
import torch.nn.functional as F
from utils import load_data, CharTokenizer
from model import TinyTransformer

BATCH_SIZE = 64
BLOCK_SIZE = 128
EPOCHS = 3500
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

text = load_data("dataset.txt")
tokenizer = CharTokenizer(text)

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

def get_batch():
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

model = TinyTransformer(
    vocab_size=tokenizer.vocab_size, 
    embed_dim=128,  
    num_layers=6     
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    x, y = get_batch()

    logits = model(x)

    loss = F.cross_entropy(
        logits.view(-1, tokenizer.vocab_size),
        y.view(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

def generate(model, start="T", length=100):
    model.eval()
    x = torch.tensor([tokenizer.encode(start)], dtype=torch.long).to(DEVICE)

    for _ in range(length):
        x_cond = x[:, -BLOCK_SIZE:] 
        
        logits = model(x_cond)
        
        temperature = 0.8 
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)

        top_k = 10
        values, indices = torch.topk(probs, top_k)
        probs_filtered = torch.zeros_like(probs).scatter_(1, indices, values)
        probs_filtered = probs_filtered / probs_filtered.sum(dim=-1, keepdim=True)

        next_token = torch.multinomial(probs_filtered, num_samples=1)

        x = torch.cat([x, next_token], dim=1)
        
    return tokenizer.decode(x[0].tolist())

print("\nGenerated Text:\n")
print(generate(model, start="T"))

torch.save(model.state_dict(), "shakespeare_model.pth")
print("\nModel saved successfully to shakespeare_model.pth!")