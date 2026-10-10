# KV cache from scratch

The code from the video "LLM Inference and KV Cache From Scratch".

A small GPT writes Shakespeare one character at a time. Seven steps measure what the model keeps
between two characters, why that makes writing fast, what it costs in memory, and when two requests can share it.

Everything runs on a laptop CPU. The only package is PyTorch.

## Run it

```bash
pip install torch
sh run_all.sh
```

The first run trains the model (about ten minutes on a laptop CPU) and saves it in `files/`.
Later runs load it. `demo-output.txt` is the output of one full run.

Run one step alone:

```bash
python step3_cache.py
```

## The steps

| File | What it does | Measured in `demo-output.txt` |
|---|---|---|
| `gpt.py` | The model, the training loop, slow writing and writing with a cache | 1,237,057 numbers, 6 layers, a window of 256 characters |
| `step1_slow.py` | Writes the slow way: the whole text goes through the model for every new character | character 16 takes 0.74 ms, character 256 takes 2.79 ms |
| `step2_same.py` | Runs the model on a text, adds one character, runs again, compares | 600 of 606 lists were already known (99.0 %) |
| `step3_cache.py` | Keeps the key and the value of every character in every layer | 3,840 of 3,840 characters the same as the slow way |
| `step4_race.py` | Races slow against cache | 256 characters: 420 ms slow, 111 ms with cache (3.8 times faster) |
| `step5_read.py` | Reads a prompt in one pass, then writes | first character after 2.3 ms, every later one after 0.46 ms |
| `step6_price.py` | Counts the bytes of the cache | 6,144 bytes per character; a full window is 32 % of the model |
| `step7_share.py` | Two requests with the same start share one cache | first character 2.5 times sooner; one changed character breaks it |
| `exercise.py` | The exercise from the end of the video | see below |
| `tests.py` | Nine checks on the cache | 9 checks passed |

## What the cache is

In every layer, attention makes three lists for every character: a query, a key and a value.
A character only looks back, so a new character at the end cannot change the keys and values of the characters before it.
The cache stores those keys and values. For a new character the model computes one query, one key and one value,
and looks the rest up.

Size of the cache for one character:

```
2 (key and value) x 6 layers x 128 numbers x 4 bytes = 6,144 bytes
```

## The exercise

The window is full at 256 characters. To write on, the oldest character has to leave.
Can we drop its key and value from the cache and keep the rest?

Write down your guess, then run:

```bash
python exercise.py
```

<details>
<summary>Answer</summary>

No. In 100 of 150 steps the favourite next character is different, and the scores are off by up to 18.19.

Every kept key and value was computed for a character at its old position, and with the dropped characters in view.
After the slide each character sits one position earlier, so the kept numbers are stale.

Two repairs, both in `exercise.py`:

- Read the whole window again for every character: 2.77 ms per character.
- Cut the text to its last 128 characters, read them once, write on with the cache: 0.53 ms per character.
  The model then sees less of the old text, so it writes a different continuation.

</details>

## Limits

- The model has 1.2 million numbers. The models you chat with have billions.
- It reads characters. Real models read tokens, which are pieces of words.
- All times are from one laptop CPU with 4 threads. Your numbers will differ.
- What carries over is the shape of the results: time per character climbs without a cache and stays flat with one.

## The same ideas in real tools

- Transformers library: the cache is the argument `past_key_values`.
- vLLM: automatic prefix caching shares the cache between requests with the same start.
- Anthropic API: prompt caching; text read from a cache costs 0.1 times the base input price, and the cache is kept 5 minutes by default.
- Llama 3 8B: 32 query heads share 8 key and value heads (grouped-query attention, arXiv 2305.13245), so the cache is 4 times smaller.

## License

MIT
