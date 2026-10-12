# Day 1 — Someone else's model

> **By the end of today** you have opened a real pretrained model and found, inside it,
> the thing you built last week.

For two weeks you have trained models from nothing: random numbers in, Shakespeare-ish
out. Nobody does that for real work. They start from a model somebody else already
trained, for weeks, on a great deal of text.

That is all "pretrained" means. **Somebody ran your loop for a long time.** The file
they ended up with is a `state_dict`, like the one you saved on Thursday, only larger.

Today you open one. The point of the day is the moment of recognition.

`render_chatml` goes in `chat.py`, in this folder. `AsCourseModel` and `layer_table`
go in `tunelab/pretrained.py`.

---

## Read first

- [ ] The [model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) for the course's pretrained model — the summary and "How to use" · 10 min
- [ ] [Chat templates](https://huggingface.co/docs/transformers/chat_templating) — the introduction, up to the first example's output · 10 min

---

## The model

`HuggingFaceTB/SmolLM2-135M-Instruct`. About 135 million parameters: seventy times
your big GPT, and roughly a thousandth of the models you call through an API. Small
enough to train on a free GPU, real enough that it writes English and follows
instructions.

You do not need to download it today. Every test this week runs on your laptop against
a miniature with the same architecture and random weights — same parts, same names,
a few thousand parameters. Friday's notebook loads the real one.

To see it, in a Python shell:

```python
from transformers import AutoModelForCausalLM, LlamaConfig

tiny = AutoModelForCausalLM.from_config(LlamaConfig(
    vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
))
print(tiny)
```

Do that now, and keep the output next to your block diagram.

### What you will recognise

| In the printout | What you called it |
|---|---|
| `embed_tokens` | the token table in `Embedder` |
| `layers` — a list of identical blocks | `blocks` |
| `self_attn` with `q_proj`, `k_proj`, `v_proj` | `query`, `key`, `value` in your heads |
| `o_proj` | the mixing layer at the end of `MultiHead` |
| `mlp` | the MLP: out wide, non-linear, back |
| `input_layernorm`, `post_attention_layernorm` | `norm1`, `norm2` — before each branch, as in yours |
| `norm` | the final norm |
| `lm_head` | the final `Linear(n_embd, vocab_size)` |

`hidden_size` is your `n_embd`. `num_hidden_layers` is your `n_layer`. It is your
diagram.

### What is different, honestly

Four things. You need one sentence about each this week, not a lecture.

- **No position table.** Position is mixed into the queries and keys by rotating them
  ("rotary" embeddings) rather than added at the start. Same job; no `block_size`-row
  table to run out of.
- **A different norm.** `RMSNorm` rescales without shifting. A cheaper cousin of
  `LayerNorm`.
- **All the heads live in one matrix.** `q_proj` is a single `Linear` that produces
  every head's query at once; the code slices it into heads afterwards. You built
  heads as separate objects because it is clearer. This is the same arithmetic,
  packed.
- **A three-layer MLP with a gate.** Still "out wide, non-linear, back".

None of these changes what you will do to it this week.

### Where 135 million parameters are

Your first exercise builds the table that answers this. For the real model, the
headline numbers:

| | Parameters | Share |
|---|---|---|
| Token table, 49,152 tokens × 576 | 28.3 M | 21% |
| 30 blocks × 3.5 M | 106.2 M | 79% |
| of which the MLPs | 79.6 M | 59% |
| of which attention | 26.5 M | 20% |

(`lm_head` is not in the table because it **shares** its weights with the token table:
the same 28 million numbers are used to read tokens in and to score them on the way
out.)

Compare the shares with what you found for your own GPT on Friday.

---

## What you need to know

### Subword tokens

Your tokenizer had 65 entries, one per character. This one has 49,152, and an entry
is usually a piece of a word:

```
"Fine-tuning"   ->   ["Fine", "-", "t", "uning"]      (roughly; the pieces vary)
```

It is the same idea as your table — text to whole numbers and back — built by a
procedure that merges the commonest neighbouring pieces until the table is full. You
use it as given. What matters this week: **a pretrained model only understands its own
tokenizer.** The numbers it learned are attached to those 49,152 rows and no others.

### A chat is one long string

You have sent models lists of messages for months:

```python
[{"role": "user", "content": "Hello"}]
```

The model never sees a list. It sees one string of text and predicts what comes next,
exactly like your GPT. Something has to flatten the list into a string, with markers
for who is speaking:

```
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
```

and then the model writes the reply, ending it with `<|im_end|>` of its own accord.

That format is called ChatML. The markers are not magic: they are single tokens in the
vocabulary, and the model was **trained** on text laid out exactly this way. Lay a
chat out any other way — a missing newline, a different marker — and you are showing
it text it has never seen the like of. It will still answer. It will answer worse,
and nothing will tell you why.

That is the most common silent bug in fine-tuning, and it is the reason you are
writing the flattening by hand once: so that in week 5, when a library does it for
you, you know there is a string underneath and you know how to look at it.

The last line of the example above — the assistant marker with nothing after it — is
the **generation prompt**: it tells the model whose turn it is. You include it when
you want a reply and leave it off when the text already contains one.

### Making it fit your loop

Your loop expects `model(x, targets)` to return `(output, loss)`. A Hugging Face model
is called differently and returns an object:

```python
out = tiny(input_ids=idx)
out.logits.shape            # (B, T, vocab_size)
```

So you write a small wrapper: an `nn.Module` that holds the Hugging Face model, calls
it that way, and hands back the pair your loop wants. When a module is stored as an
attribute of another module (`self.something = hf_model`), PyTorch can see inside it:
its parameters show up in `.parameters()`, and `.to(device)` and `.train()` reach it.

**The trap.** Hugging Face models will compute a loss for you if you pass `labels=`.
Do not. When given labels they assume you passed *the same tokens as the input* and
shift them by one position internally. Your batches are already shifted — `y` is `x`,
one position later — because you wrote `make_get_batch` that way in week 1. Shift
twice and the model is trained to predict the token after next. It runs. The loss goes
down. The model is quietly wrong.

So take the logits and compute the loss yourself, the way your GPT does. One of
today's tests checks for exactly this.

---

## Exercises

1. `layer_table(model)` in `tunelab/pretrained.py`. `model.named_parameters()` yields
   `(name, tensor)` pairs.
2. `AsCourseModel` in `tunelab/pretrained.py`.
3. `render_chatml` in `chat.py`.

```bash
pytest week-03/day-1 -v
```

One test trains the wrapped model with `tunelab/loop.py`, the file you wrote on the
third day of the course and have not edited since. When it passes, stop and notice
what just happened.

---

## Before tomorrow

Add today's row to `logs/signal-log.md`.

Then, on paper: tomorrow you fine-tune this model the obvious way — every one of its
135 million parameters trainable. AdamW keeps two extra numbers for every parameter it
trains, and backward needs a gradient for each. **How many numbers is that in memory,
before the model has looked at a single token?**
