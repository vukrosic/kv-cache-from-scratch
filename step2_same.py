"""Step 2. Read 100 characters, then 101. What changed inside the model for the first 100?"""
from gpt import *

model = trained()
ids = HELD[:101].unsqueeze(0)
a, b = [], []
with torch.no_grad():
    model(ids[:, :100], trace=a)                # the lists after every layer, 100 characters
    model(ids, trace=b)                         # the same text with one more character

print('Largest change in any number of the first 100 characters:')
diffs, sizes = [], []
for i in range(LAYERS):
    diffs.append((a[i] - b[i][:, :100]).abs().max().item())
    sizes.append(a[i].abs().mean().item())
    print(f'  after layer {i + 1}: {diffs[-1]:.7f}   (a typical number there is {sizes[-1]:.2f})')
lists = 101 * LAYERS
print(f'\nThe second run computed {lists} lists. {100 * LAYERS} of them were already known: {100 * 100 * LAYERS / lists:.1f} %.')
print('Reason: a character only looks back. A new character at the end cannot change the ones before it.')
save('same', diffs=diffs, sizes=sizes, largest=max(diffs), lists=lists, known=100 * LAYERS, share=100 * 100 * LAYERS / lists)
