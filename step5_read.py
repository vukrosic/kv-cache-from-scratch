"""Step 5. Reading and writing are two different jobs."""
from gpt import *

model = trained()
ids = HELD[:200].unsqueeze(0)                   # a prompt of 200 characters


@torch.no_grad()
def together():
    kept = new_cache()
    model(ids, kept)                            # all 200 characters in one pass
    return kept


@torch.no_grad()
def one_by_one():
    kept = new_cache()
    for i in range(200):
        model(ids[:, i:i + 1], kept)            # the same 200 characters, one after the other
    return kept


a, b = together(), one_by_one()
gap = max((x[t] - y[t]).abs().max().item() for x, y in zip(a['layers'], b['layers']) for t in 'kv')
t_all, t_one = best(together) * 1000, best(one_by_one) * 1000
print(f'Reading 200 characters in one pass:      {t_all:6.1f} ms  ({t_all / 200:.3f} ms per character)')
print(f'Reading them one after the other:        {t_one:6.1f} ms  ({t_one / 200:.3f} ms per character)')
print(f'Both leave the same cache (largest gap {gap:.6f}). One pass is {t_one / t_all:.1f} times faster.')

clock = []
for _ in range(15):
    c = []
    write_fast(model, decode(ids[0]), 50, clock=c)
    clock.append(c)
first = min(c[0] for c in clock) * 1000
later = sum(min(c[i] for c in clock) for i in range(1, 50)) / 49 * 1000
print(f'\nA request with this prompt: the first character arrives after {first:.1f} ms, every later one after {later:.2f} ms.')
print('Reading the prompt is called prefill. Writing one character at a time is called decode.')
save('read', together=t_all, one=t_one, ratio=t_one / t_all, gap=gap, first=first, later=later)
