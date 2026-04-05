import torch
import torch.nn.functional as F
from utils import load_data, CharTokenizer
from model import TinyTransformer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = 64 

text = load_data("dataset.txt")
tokenizer = CharTokenizer(text)

model = TinyTransformer(vocab_size=tokenizer.vocab_size).to(DEVICE)
model.load_state_dict(torch.load("shakespeare_model.pth", weights_only=True))
model.eval() 

print("Model loaded successfully!\n")

def generate_text(model, start="O Romeo", length=200, temperature=0.8):
    x = torch.tensor([tokenizer.encode(start)], dtype=torch.long).to(DEVICE)

    for _ in range(length):
        x_cond = x[:, -BLOCK_SIZE:]
        
        logits = model(x_cond)
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)

        top_k = 10
        values, indices = torch.topk(probs, top_k)
        probs_filtered = torch.zeros_like(probs).scatter_(1, indices, values)
        probs_filtered = probs_filtered / probs_filtered.sum(dim=-1, keepdim=True)

        next_token = torch.multinomial(probs_filtered, num_samples=1)
        x = torch.cat([x, next_token], dim=1)
        
    return tokenizer.decode(x[0].tolist())

print("Generated Text:")
print("-" * 30)
print(generate_text(model, start="O ", length=500, temperature=0.8))