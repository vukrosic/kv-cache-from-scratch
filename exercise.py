"""Exercise. The window is full: 256 characters. To go on, the oldest character has to leave.

Can we drop the oldest character from the cache and keep writing with the rest?
Write down your guess, then run this file.
"""
from gpt import *

model = trained()
MORE = 150                                      # characters to write after the window is full
start = decode(HELD[:BLOCK])


@torch.no_grad()
def slide_right(n):
    """The correct way: for every new character read the last 256 characters again."""
    idx = torch.tensor([encode(start)])
    scores = []
    for _ in range(n):
        scores.append(model(idx[:, -BLOCK:])[:, -1])
        idx = torch.cat([idx, scores[-1].argmax(-1, keepdim=True)], 1)
    return decode(idx[0, BLOCK:]), torch.cat(scores)


@torch.no_grad()
def slide_cache(n, follow):
    """Drop the oldest keys and values, feed only the new character. follow: the correct text, fed in at every step."""
    kept = new_cache()
    scores = [model(torch.tensor([encode(start)]), kept)[:, -1]]
    for ch in follow[:n - 1]:
        for L in kept['layers']:
            L['k'], L['v'] = L['k'][:, :, 1:], L['v'][:, :, 1:]     # the oldest character leaves
        kept['n'] = BLOCK - 1
        scores.append(model(torch.tensor([encode(ch)]), kept)[:, -1])
    return torch.cat(scores)


right, s_right = slide_right(MORE)
s_cache = slide_cache(MORE, right)
wrong = (s_right.argmax(-1) != s_cache.argmax(-1)).sum().item()
gap = (s_right - s_cache).abs().max().item()
print(f'{MORE} characters past the window, each time with the correct text before it:')
print(f'  the favourite next character is different in {wrong} of {MORE} steps')
print(f'  the scores are off by up to {gap:.2f}')
print('Why: every kept key and value was computed for a character at its old position, and with the dropped')
print('characters in view. After the slide each character sits one position earlier, so the kept numbers are stale.')

t_right = best(lambda: slide_right(20), 7) / 20 * 1000


@torch.no_grad()
def reread(n):
    """A common repair: when the window is full, cut the text to its last half and read that again once."""
    text, kept, out = start, None, ''
    while len(out) < n:
        text = text[-BLOCK // 2:]
        m = min(BLOCK - len(text), n - len(out))
        new = write_fast(model, text, m)
        out, text = out + new, text + new
    return out


t_reread = best(lambda: reread(MORE), 7) / MORE * 1000
print(f'\nRead the whole window again for every character: {t_right:.2f} ms per character.')
print(f'Cut to the last {BLOCK // 2} characters, read them once, write on with the cache: {t_reread:.2f} ms per character.')
print('The second way is fast, but the model now sees less of the old text, so it writes a different continuation.')
save('exercise', wrong=wrong, more=MORE, gap=gap, right=t_right, reread=t_reread)
