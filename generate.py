import torch
import torch.nn.functional as F
from utils import load_data, CharTokenizer
from model import TinyTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = 128 

text = load_data("dataset.txt")
tokenizer = CharTokenizer(text)

model = TinyTransformer(
    vocab_size=tokenizer.vocab_size, 
    embed_dim=128, 
    num_layers=6
).to(DEVICE)

model.load_state_dict(torch.load("shakespeare_model.pth", weights_only=True))
model.eval() 

print("\nModel loaded successfully! It's chat time.\n")

def generate_text(model, start="To be, or not", length=200, temperature=0.8, top_p=0.9):
    x = torch.tensor([tokenizer.encode(start)], dtype=torch.long).to(DEVICE)

    for _ in range(length):
        x_cond = x[:, -BLOCK_SIZE:]
        
        logits = model(x_cond)
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)

        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0
        
        probs_sort = probs_sort / probs_sort.sum(dim=-1, keepdim=True)
        
        next_token_idx = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token_idx)

        x = torch.cat([x, next_token], dim=1)
        
    return tokenizer.decode(x[0].tolist())

print("=" * 50)
print(" Welcome to the AI Shakespeare Bot!")
print(" Type 'quit' to exit.")
print("=" * 50)

while True:
    user_input = input("\nYou: ")
    
    # FIX 1: Added .strip() so accidental trailing spaces don't break the quit command!
    if user_input.strip().lower() == 'quit':
        print("Farewell!")
        break
    
    # FIX 2: Simplified the prompt. 
    # Your old code did `prompt = f"\n{user_input.upper()}:\n"`
    # If you typed "hi", it fed the AI "\nHI:\n". Shakespeare doesn't use all-caps 
    # words with colons like that, so the AI got confused and stayed silent.
    prompt = user_input.strip() + " "
    
    # FIX 3: Lowered length to 150. Generating 400 characters one-by-one is what 
    # caused your terminal to feel completely frozen while it was "thinking".
    response = generate_text(model, start=prompt, length=150, temperature=0.8, top_p=0.9)
    
    print(f"\nAI: {response[len(prompt):]}")