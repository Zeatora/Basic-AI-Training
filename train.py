import torch
import torch.nn.functional as F
from utils import load_data, CharTokenizer
from model import TinyTransformer

BATCH_SIZE = 16
BLOCK_SIZE = 32
EPOCHS = 200
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

text = load_data("dataset.txt")
tokenizer = CharTokenizer(text)

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

def get_batch():
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

model = TinyTransformer(vocab_size=tokenizer.vocab_size).to(DEVICE)
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

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

def generate(model, start="T", length=100):
    model.eval()
    x = torch.tensor([tokenizer.encode(start)], dtype=torch.long).to(DEVICE)

    for _ in range(length):
        logits = model(x)
        temperature = 1.2
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)

        if x.shape[1] > 5:
            last_token = x[0, -1]
            probs[0, last_token] *= 0.3

            
        top_k = 10
        values, indices = torch.topk(probs, top_k)
        probs_filtered = torch.zeros_like(probs).scatter_(1, indices, values)
        probs_filtered = probs_filtered / probs_filtered.sum(dim=-1, keepdim=True)

        

        next_token = torch.multinomial(probs_filtered, num_samples=1)

        x = torch.cat([x, next_token], dim=1)
        

    return tokenizer.decode(x[0].tolist())

print("\nGenerated Text:\n")
print(generate(model, start="T"))