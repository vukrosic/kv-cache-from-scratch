"""Checks. Run: python tests.py"""
from gpt import *

model = trained()
ids = HELD[:BLOCK].unsqueeze(0)
done = 0


def check(name, ok):
    global done
    assert ok, name
    done += 1
    print('ok  ', name)


with torch.no_grad():
    full = model(ids)
    kept = new_cache()
    steps = torch.cat([model(ids[:, i:i + 1], kept) for i in range(BLOCK)], 1)
    half = new_cache()
    two = torch.cat([model(ids[:, :100], half), model(ids[:, 100:], half)], 1)
    a, b = [], []
    model(ids[:, :100], trace=a)
    model(ids[:, :101], trace=b)

check('one character at a time gives the scores of one pass', (full - steps).abs().max() < 1e-3)
check('reading in two pieces gives the scores of one pass', (full - two).abs().max() < 1e-3)
check('the cache holds every character once', kept['n'] == BLOCK and kept['layers'][0]['k'].shape == (1, HEADS, BLOCK, EMB // HEADS))
check('cache size follows the rule', cache_bytes(kept) == 2 * LAYERS * BLOCK * HEADS * (EMB // HEADS) * 4)
check('a new character does not change the ones before it', max((x - y[:, :100]).abs().max() for x, y in zip(a, b)) < 1e-4)
check('slow and cached writing agree, favourite', write_slow(model, PROMPT, 120) == write_fast(model, PROMPT, 120))
check('slow and cached writing agree, loaded die', write_slow(model, PROMPT, 120, seed=5) == write_fast(model, PROMPT, 120, seed=5))
cut = copy_cache(kept, upto=50)
check('a cut copy keeps only the first characters', cut['n'] == 50 and cut['layers'][-1]['v'].shape[2] == 50)
with torch.no_grad():
    again = model(ids[:, 50:], cut)
check('a cut cache can be filled again', (full[:, 50:] - again).abs().max() < 1e-3)
print(f'\n{done} checks passed')
