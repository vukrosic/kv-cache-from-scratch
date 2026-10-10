"""A small GPT in PyTorch and two ways to let it write: the slow way and the way with a KV cache.

Every step file imports this. The model is the one from "Build and Train Your Own GPT From Scratch",
with a longer window: it can look back over 256 characters.
"""
import os, re, json, math, time, hashlib
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.join(HERE, 'files')

# ---------------------------------------------------------------- the text
TEXT = open(os.path.join(HERE, 'shakespeare.txt')).read()
CHARS = sorted(set(TEXT))                       # every different character, 65 of them
V = len(CHARS)
STOI = {c: i for i, c in enumerate(CHARS)}


def encode(s):
    return [STOI[c] for c in s]


def decode(ids):
    return ''.join(CHARS[int(i)] for i in ids)


DATA = torch.tensor(encode(TEXT))
CUT = int(len(DATA) * 0.9)
TRAIN, HELD = DATA[:CUT], DATA[CUT:]            # the model never trains on HELD
REAL = set(re.findall(r"[a-z']+", TEXT.lower()))

BLOCK, EMB, HEADS, LAYERS = 256, 128, 4, 6      # window, numbers per character, heads, layers


# ---------------------------------------------------------------- the model
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(EMB), nn.LayerNorm(EMB)
        self.query = nn.Linear(EMB, EMB, bias=False)    # what am I looking for?
        self.key = nn.Linear(EMB, EMB, bias=False)      # what do I hold?
        self.value = nn.Linear(EMB, EMB, bias=False)    # what do I hand over?
        self.out = nn.Linear(EMB, EMB)
        self.think = nn.Sequential(nn.Linear(EMB, 4 * EMB), nn.GELU(), nn.Linear(4 * EMB, EMB))

    def mix(self, x, kept=None):
        """Attention. x holds only the new characters when kept (the KV cache of this layer) is given."""
        B, T, _ = x.shape
        heads = lambda t: t.view(B, T, HEADS, EMB // HEADS).transpose(1, 2)     # B, heads, T, 32
        q, k, v = heads(self.query(x)), heads(self.key(x)), heads(self.value(x))
        if kept is not None:
            if 'k' in kept:                             # old keys and values come from the cache
                k = torch.cat([kept['k'], k], 2)
                v = torch.cat([kept['v'], v], 2)
            kept['k'], kept['v'] = k, v                 # and the new ones go in
        match = q @ k.transpose(-1, -2) / math.sqrt(EMB // HEADS)               # B, heads, new, all
        all_ = k.shape[2]
        future = torch.triu(torch.ones(T, all_, dtype=torch.bool), all_ - T + 1)
        share = match.masked_fill(future, float('-inf')).softmax(-1)            # each row adds up to 1
        return self.out((share @ v).transpose(1, 2).reshape(B, T, EMB))

    def forward(self, x, kept=None):
        x = x + self.mix(self.norm1(x), kept)           # look at the other characters
        return x + self.think(self.norm2(x))            # think about what was found


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.letter = nn.Embedding(V, EMB)              # a list of numbers for each character
        self.place = nn.Embedding(BLOCK, EMB)           # a list of numbers for each position
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.norm = nn.LayerNorm(EMB)
        self.head = nn.Linear(EMB, V)                   # back to 65 scores

    def forward(self, idx, kept=None, trace=None):
        """Scores for the next character at every position of idx.

        kept: a KV cache from new_cache(). Then idx holds only the characters the cache has not seen.
        trace: a list; the lists of numbers after every layer are added to it.
        """
        seen = kept['n'] if kept else 0
        x = self.letter(idx) + self.place(torch.arange(seen, seen + idx.shape[1]))
        for i, b in enumerate(self.blocks):
            x = b(x, kept['layers'][i] if kept else None)
            if trace is not None:
                trace.append(x)
        if kept:
            kept['n'] = seen + idx.shape[1]
        return self.head(self.norm(x))


def new_cache():
    return {'n': 0, 'layers': [{} for _ in range(LAYERS)]}


def copy_cache(kept, upto=None):
    """A copy of a cache, cut to its first upto characters."""
    n = kept['n'] if upto is None else upto
    return {'n': n, 'layers': [{'k': L['k'][:, :, :n].clone(), 'v': L['v'][:, :, :n].clone()} for L in kept['layers']]}


def cache_bytes(kept):
    return sum(t.numel() * t.element_size() for L in kept['layers'] for t in L.values())


def count(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------- two ways to write
def pick(scores, gen):
    """The next character from 65 scores: the favourite, or a roll of a loaded die when gen is given."""
    if gen is None:
        return scores.argmax(-1, keepdim=True)
    return torch.multinomial(scores.softmax(-1), 1, generator=gen)


@torch.no_grad()
def write_slow(model, start, n, seed=None, clock=None):
    """Read the whole text again for every new character."""
    gen = None if seed is None else torch.Generator().manual_seed(seed)
    idx = torch.tensor([encode(start)])
    for _ in range(n):
        t0 = time.perf_counter()
        scores = model(idx)[:, -1]                      # all characters go through all layers
        idx = torch.cat([idx, pick(scores, gen)], 1)
        if clock is not None:
            clock.append(time.perf_counter() - t0)
    return decode(idx[0, len(start):])


@torch.no_grad()
def write_fast(model, start, n, seed=None, clock=None, kept=None, after=None):
    """Read the start once, keep the keys and values, then feed one new character at a time.

    kept: a cache that already holds the first kept['n'] characters of start.
    after: called with the cache after every step (step 6 uses it to round the kept numbers).
    """
    gen = None if seed is None else torch.Generator().manual_seed(seed)
    kept = kept or new_cache()
    new = torch.tensor([encode(start)])[:, kept['n']:]
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        scores = model(new, kept)[:, -1]                # only the new characters go through the layers
        new = pick(scores, gen)
        if after:
            after(kept)
        out.append(new)
        if clock is not None:
            clock.append(time.perf_counter() - t0)
    return decode(torch.cat(out, 1)[0])


# ---------------------------------------------------------------- measuring
@torch.no_grad()
def score(model, data=HELD, pieces=100):
    """Loss and the share of next characters guessed right, on evenly spaced pieces of unseen text."""
    at = torch.linspace(0, len(data) - BLOCK - 2, pieces).long()
    x = torch.stack([data[i:i + BLOCK] for i in at])
    y = torch.stack([data[i + 1:i + BLOCK + 1] for i in at])
    out = model(x)
    return F.cross_entropy(out.view(-1, V), y.view(-1)).item(), (out.argmax(-1) == y).float().mean().item() * 100


def real_words(text):
    """Of the words with three letters or more, the share that appears somewhere in the plays."""
    words = [w for w in re.findall(r"[a-z']+", text.lower()) if len(w) >= 3]
    return 100 * sum(w in REAL for w in words) / max(len(words), 1)


def best(fn, repeats=15):
    """Run fn several times and return the shortest time in seconds (the run least disturbed by other programs)."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def clocks(write, model, n, runs=9):
    """Milliseconds for each of n new characters: the best of several runs. near() reads one place from it."""
    all_ = []
    for _ in range(runs):
        clock = []
        write(model, PROMPT, n, clock=clock)
        all_.append(clock)
    return [min(r[i] for r in all_) * 1000 for i in range(n)]


def near(per, place):
    """The time at a place in the text: the middle value of the nine characters that end there."""
    i = place - len(PROMPT) - 1
    return sorted(per[i - 8:i + 1])[4]


# ---------------------------------------------------------------- training
ROUNDS, SIZE = 3000, 32


def trained():
    """Train once (about ten minutes on a laptop CPU), save in files/, load on later runs."""
    os.makedirs(FILES, exist_ok=True)
    tag = hashlib.md5(json.dumps([BLOCK, EMB, HEADS, LAYERS, ROUNDS, SIZE]).encode()).hexdigest()[:10]
    path = os.path.join(FILES, f'gpt-{tag}.pt')
    torch.manual_seed(0)
    model = GPT()
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        return model.eval()
    gen = torch.Generator().manual_seed(0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(ROUNDS):
        at = torch.randint(len(TRAIN) - BLOCK - 1, (SIZE,), generator=gen)
        x = torch.stack([TRAIN[i:i + BLOCK] for i in at])
        y = torch.stack([TRAIN[i + 1:i + BLOCK + 1] for i in at])
        loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0:
            print(f'  round {step:,}: miss {loss.item():.2f}', flush=True)
    torch.save(model.state_dict(), path)
    return model.eval()


def save(step, **numbers):
    """Keep the measured numbers of a step in files/, so the video and the README quote the same run."""
    os.makedirs(FILES, exist_ok=True)
    json.dump(numbers, open(os.path.join(FILES, f'values-{step}.json'), 'w'), indent=1)


def load(step):
    return json.load(open(os.path.join(FILES, f'values-{step}.json')))


def starts(n=8, length=16):
    """n short pieces of unseen text to start writing from."""
    return [decode(HELD[i * 5000:i * 5000 + length]) for i in range(n)]


PROMPT = 'ROMEO:\n'      # every step starts its text with this unless it says otherwise
