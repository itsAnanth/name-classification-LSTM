# %%
import torch
from data import Data
from train import train
from LSTM import LSTM
import matplotlib.pyplot as plt



# %%

data = Data()

model = LSTM(len(data.vocab), 128, len(data.categories))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
losses = []



# %%
train(
    model,
    criterion,
    optimizer,
    epochs=100000,
    data=data,
    losses=losses
)

# %%
w, c = data.random_sample()

c

# %%
def predict(model, input, data):
    tensor = data.word2tensor(input)
    hidden_state = model.init_hidden()

    for ch in range(tensor.shape[0]):
        output, hidden_state = model(tensor[ch], hidden_state)

    category = data.tensor2category(torch.tensor([torch.argmax(output)], dtype=torch.long))
    return category

# predict(model, "Luka Modric\n", data)


# %%
plt.plot(range(len(losses)), losses)

# %%
while True:
    inp = input("> ")
    if inp.lower() == "quit":
        break

    category = predict(model, f"{inp}\n", data)
    print(category)


