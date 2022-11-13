import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

category_lines, all_categories = load_data()
n_categories = len(all_categories)
n_letters = len(all_letters)
n_hidden = 128

rnn = RNN(n_letters, n_hidden, n_categories)
rnn.to(DEVICE)

def categoryFromOutput(output):
    category_idx= torch.argmax(output).item()
    return all_categories[category_idx]

print(categoryFromOutput(rnn(lineToTensor('Albert')[0], rnn.initHidden())[0]))

criteria = nn.NLLLoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=0.005)

def train(line_tensor, category_tensor):
    line_tensor = line_tensor.to(DEVICE)
    category_tensor = category_tensor.to(DEVICE)

    hidden = rnn.initHidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criteria(output, category_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()

current_loss = 0
all_losses = []
plot_steps, print_steps = 1000, 5000
n_iters = 100000

for iter in range(n_iters):
    category, line, category_tensor, line_tensor = get_random_training_example(category_lines, all_categories)
    output, loss = train(line_tensor, category_tensor)
    current_loss += loss

    if (iter + 1) % plot_steps == 0:
        all_losses.append(current_loss / plot_steps)
        current_loss = 0

    if (iter + 1) % print_steps == 0:
        guess = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print(f"Iteration: {iter + 1} Loss: {loss:.4f} Name: {line} / {guess} {correct}")

plt.figure()
plt.plot(all_losses)
plt.show()

# Save model
torch.save(rnn.state_dict(), 'rnn.pth')

def predict(input_line):
    print('Predicting name: ', input_line)
    with torch.no_grad():
        line_tensor = lineToTensor(input_line)
        hidden = rnn.initHidden()
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        guess = categoryFromOutput(output)
        print('Predicted category: ', guess)

while True:
    name = input('Enter name: ')
    if name == 'q':
        break
    predict(name)