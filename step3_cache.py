"""Step 3. Keep the keys and values of every layer. Does the model still write the same text?"""
from gpt import *

model = trained()
n = BLOCK - 16
same = total = 0
for seed, start in enumerate(starts()):
    for s in (None, seed):                      # once always the favourite, once with the loaded die
        slow, fast = write_slow(model, start, n, seed=s), write_fast(model, start, n, seed=s)
        same += sum(x == y for x, y in zip(slow, fast))
        total += n
print(f'Characters written both ways: {total:,}. The same in both: {same:,}.')

ids = HELD[:BLOCK].unsqueeze(0)
with torch.no_grad():
    full = model(ids)                           # one pass over 256 characters
    kept = new_cache()
    step = torch.cat([model(ids[:, i:i + 1], kept) for i in range(BLOCK)], 1)   # one character at a time
gap = (full - step).abs().max().item()
print(f'Largest gap between the scores of the two ways: {gap:.6f} (a typical score is {full.abs().mean().item():.1f}).')

k = kept['layers'][0]['k']
print(f'\nThe cache after {BLOCK} characters: {LAYERS} layers, each with keys and values of shape {tuple(k.shape[1:])}')
print(f'(heads, characters, numbers per head). That is {cache_bytes(kept) // 4:,} numbers.')
save('cache', same=same, total=total, gap=gap, typical=full.abs().mean().item(), kept=cache_bytes(kept) // 4)
