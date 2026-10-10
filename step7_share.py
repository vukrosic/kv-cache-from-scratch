"""Step 7. Two requests start with the same text. Read that part once and share its cache."""
from gpt import *

model = trained()
RULES = TEXT[:200]                              # the shared start: 200 characters both requests begin with
one, two = RULES + 'ROMEO:\n', RULES + 'JULIET:\n'


@torch.no_grad()
def read(text, kept=None):
    """Prefill: put text into a cache (the part the cache already holds is skipped)."""
    kept = kept or new_cache()
    model(torch.tensor([encode(text)])[:, kept['n']:], kept)
    return kept


shared = copy_cache(read(one), upto=len(RULES))     # request one was read; keep the cache of the shared start

fresh = write_fast(model, two, 40)
reused = write_fast(model, two, 40, kept=copy_cache(shared))
t_fresh = best(lambda: read(two)) * 1000
t_reuse = best(lambda: read(two, copy_cache(shared))) * 1000
print(f'Request two is {len(two)} characters long. {len(RULES)} of them are the shared start.')
print(f'Read all of it:                {t_fresh:5.2f} ms before the first character.')
print(f'Take the shared cache instead: {t_reuse:5.2f} ms. That is {t_fresh / t_reuse:.1f} times sooner.')
print(f'The 40 characters it writes are the same: {fresh == reused}.')

# now change one character near the beginning of the shared start
at = 5
three = RULES[:at] + ('x' if RULES[at] != 'x' else 'y') + RULES[at + 1:] + 'JULIET:\n'
right = write_fast(model, three, 40)
wrong = write_fast(model, three, 40, kept=copy_cache(shared))          # pretend the old cache still fits
differ = sum(x != y for x, y in zip(right, wrong))
with torch.no_grad():
    s_right = model(torch.tensor([encode(three)]))[0, -1]
    s_wrong = model(torch.tensor([encode(three)])[:, len(RULES):], copy_cache(shared))[0, -1]
gap = (s_right - s_wrong).abs().max().item()
careful = write_fast(model, three, 40, kept=copy_cache(shared, upto=at))  # keep only what is still the same
t_careful = best(lambda: read(three, copy_cache(shared, upto=at))) * 1000
print(f'\nRequest three: character {at + 1} of the shared start is different.')
print(f'Use the old cache anyway: the scores for the next character are off by up to {gap:.2f}, '
      f'and {differ} of 40 characters come out different.')
print(f'Keep only the first {at} characters of the cache: the text is right again ({careful == right}), '
      f'but reading takes {t_careful:.2f} ms.')
print('A cache fits only a text that starts with exactly the same characters.')
save('share', length=len(two), shared=len(RULES), fresh=t_fresh, reuse=t_reuse, ratio=t_fresh / t_reuse, same=fresh == reused,
     at=at + 1, gap=gap, differ=differ, careful=t_careful, kept=at)
