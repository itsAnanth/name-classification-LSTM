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
        hidden = model.init_hidden()
        state = model.init_state()


        for ch in range(word.shape[0]):
            out, hidden, state = model(word[ch], hidden, state)

    
        loss = criterion(out, category)



        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        
        if (i % eval_len == 0):
            correct = data.tensor2category(category.item())
            guess = data.tensor2category(torch.argmax(out))

            check = '✓' if guess == correct else f'✗'
            print(f"Actual: {correct} | Predicted: {guess} {check}")
        
        if (i % eval_len == 0):
            print(f"Iterations: {i} | loss: {running_loss / eval_len}")
            losses.append(running_loss / eval_len)
            running_loss = 0
            
            
        running_loss += loss.item()
        