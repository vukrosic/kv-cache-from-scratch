"""Step 4. The race: time for each new character, the slow way against the cache."""
from gpt import *

model = trained()
n = BLOCK - len(PROMPT)
slow = load('slow')['per']                      # measured in step 1
fast = clocks(write_fast, model, n)
print('Time to write one character (ms):      slow    cache')
at = {}
for place in (16, 32, 64, 128, 256):
    at[place] = [near(slow, place), near(fast, place)]
    print(f'  character {place:>3}:                     {at[place][0]:6.2f}   {at[place][1]:5.2f}')
print('\nTime to write the text up to a length (ms):')
upto = {}
for length in (64, 128, 256):
    i = length - len(PROMPT)
    upto[length] = [sum(slow[:i]), sum(fast[:i]), sum(slow[:i]) / sum(fast[:i])]
    print(f'  {length:>3} characters: slow {upto[length][0]:6.0f}   cache {upto[length][1]:5.0f}   {upto[length][2]:.1f} times faster')
passes_slow = sum(len(PROMPT) + i for i in range(n))
passes_fast = len(PROMPT) + n - 1
print(f'\nCharacters pushed through the layers: slow {passes_slow:,}, cache {passes_fast}.')
print('The first step of the cache reads the whole start, so it is the slowest: ' + f'{fast[0]:.2f} ms.')
save('race', slow=slow, fast=fast, at=at, upto=upto, passes_slow=passes_slow, passes_fast=passes_fast)
