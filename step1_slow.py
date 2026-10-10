"""Step 1. The slow way: for every new character the model reads the whole text again."""
from gpt import *

model = trained()
loss, right = score(model)
sample = write_slow(model, PROMPT, BLOCK - len(PROMPT), seed=1)      # the window holds 256 characters
print(f'The model: {count(model):,} numbers, {LAYERS} layers. It looks back over {BLOCK} characters.')
print(f'On text it never saw it names the next character right {right:.1f} % of the time.')
print(f'Real words in what it writes: {real_words(sample):.0f} %. A sample:\n')
print(PROMPT + sample[:160] + '\n')

n = BLOCK - len(PROMPT)                         # write until the window is full
per = clocks(write_slow, model, n)             # ms for each new character

print('Time to write one character, by its place in the text:')
at = {}
for place in (16, 32, 64, 128, 256):
    at[place] = near(per, place)
    print(f'  character {place:>3}: {at[place]:5.2f} ms')
passes = sum(len(PROMPT) + i for i in range(n))
print(f'\nAll {n} characters: {sum(per):.0f} ms.')
print(f'Characters pushed through the layers: {passes:,} (the text is only {BLOCK} long).')
save('slow', numbers=count(model), right=right, words=real_words(sample), sample=PROMPT + sample[:160],
     per=per, at=at, total=sum(per), passes=passes, written=n)
