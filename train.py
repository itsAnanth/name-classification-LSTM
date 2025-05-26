import torch
import torch.utils
import torch.utils.data
from LSTM import LSTM
import torch.nn.functional as F

def train(model: LSTM, criterion, optimizer, epochs, data, losses = []):
    
    running_loss = 0
    i = 0
    eval_len = 500

    while True:
        i += 1
        optimizer.zero_grad()
        word, category = data.random_sample()

    
        ht = model.init_hidden()
        ct = model.init_state()
        
        for w in word:
            out, ht, ct = model(w, ht, ct)

        
        loss = criterion(out, category)


        loss.backward()
        
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        running_loss += loss.item()
        
        
        if (i % eval_len == 0):
            print(f"Iterations: {i} | loss: {running_loss / eval_len}")
            losses.append(running_loss / eval_len)
            running_loss = 0
            
            
        