"""Step 6. The price: the cache is memory, and it grows with every character."""
from gpt import *

model = trained()
kept = new_cache()
with torch.no_grad():
    model(HELD[:BLOCK].unsqueeze(0), kept)
measured = cache_bytes(kept)
rule = 2 * LAYERS * BLOCK * HEADS * (EMB // HEADS) * 4      # keys and values x layers x characters x heads x numbers per head x bytes
weights = count(model) * 4
print(f'One character in the cache: {measured // BLOCK:,} bytes.')
print(f'A full window of {BLOCK} characters: {measured:,} bytes. The rule gives {rule:,}.')
print(f'The model itself: {weights:,} bytes. One full chat keeps {100 * measured / weights:.0f} % of that.')
print(f'{weights / measured:.1f} full chats at the same time need as much memory as the model.')


def to_16_bits(kept):
    for L in kept['layers']:
        L['k'], L['v'] = L['k'].half().float(), L['v'].half().float()   # keep each number in 2 bytes, not 4


n = BLOCK - 16
changed = total = 0
for start in starts():
    exact, small = write_fast(model, start, n), write_fast(model, start, n, after=to_16_bits)
    changed += sum(x != y for x, y in zip(exact, small))
    total += n
print(f'\nKeep every number in 2 bytes: the cache is half the size, {measured // 2:,} bytes.')
print(f'Of {total:,} characters written (always the favourite), {changed} came out different.')
save('price', per_char=measured // BLOCK, full=measured, rule=rule, weights=weights, share=100 * measured / weights,
     chats=weights / measured, half=measured // 2, changed=changed, total=total)
