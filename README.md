<h1 align="center">
  <img src="https://raw.githubusercontent.com/rlouf/txt/master/docs/logo.png">
</h1>

<h3 align="center">
  Natural language generation made easy.
</h3>

<p align="center">
  <a href="https://github.com/rlouf/txt/actions?query=workflow%3Abuild"><img src="https://github.com/rlouf/txt/workflows/build/badge.svg?branch=master"></a>
  <a href="https://github.com/rlouf/txt/actions?query=workflow%3Alint"><img src="https://github.com/rlouf/txt/workflows/lint/badge.svg?branch=master"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

- **Practioners** can quickly experiment with different method/model
  combinations. The API is straightfoward, many pre-trained models are included.
- **Researchers** can easily plug *any* model to the methods and implement
  arbitrarily complex generation logic.

```python
from txt.models import GPT2
from txt.write import GreedySearch, Sampler

model = GPT2.from_pretrained('gpt2-medium')

writer1 = GreedySearch.using(model)
writer2 = Sampler.using(model).on("gpu").configure(k=10, p=0.5)

writer1.generate_until(".")
writer2.generate_until(".")
```

|              | GPT   | GPT2  | XLNet | Transfo-XL | CTRL  |
| :---         | :---: | :---: | :---: |  :---:     | :---: |
| GreedySearch | Yes   | Yes   | -     |  -         |  -    |
| Sampling     | Yes   | Yes   | -     |  -         |  -    |

## Philosophy

Text generation is arguably one of the most fun & useful aspects of modern
Natural Language Processing. Unfortunately for practitioners like myself, 
it is hard to find implementations that are simple to use and also flexible for
quick experimentation. `txt` fulfills that need. You can:

- Use 3 generation methods, with sensible default parameters;
- Load X pre-trained models with one line (thanks to HuggingFace);
- Straightforwardly generate text with `generate` and `generate_until`;
- But use the token ids directly if you feel like it;
- Implement advanced generation patterns.

On the research side, one can find several implementations of various generation
online. But those are often tied to the specific workflow of the person who
wrote the code, are untested or hard to understand and extend. With `txt` you
can:

- Easily plug in your model;
- Be confident the correctness of the code;
- Read and understand the source code. This is sadly very cumbersome in most of
  the librarie we know of. We went to great lengths to make the code legible.
  Nitpicking PRs regarding names and code organization are welcome.
- Easily add new functionalities. We tried to keep the code as modular as
  possible so that if you want to add say, another filter for sampling, you
  only need to add the function and modify one line in the code (please submit a
  PR so everyone can benefit from your work!)

## Advanced usage

### Advanced generation logic

`generate` and `generate_until` should be enough for 99% of use cases. If you are part of the 1%, we've got you covered too.

Under the hood, writer classes use a python generator to yield tokens when
asked. You can instantiate this generator yourself and implement your own
generation logic.

For instance, let's assume that you want to generate 3 sentences that end with
a period.

```python
tokens = writer.tokens()

period_token = 0

num_sentences = 0
sequence = []
for token in tokens:
  sequence.append(token)
  if token == period_token:
    num_sentences += 1
    if num_sentences == 3:
      break
```

### Using your own model

`txt` is batteries-included, but also allows you to bring your own model. For
this, your model class needs to implement the following method:

- `decode` which takes a list of token ids as an input and returns the logits
  for the next token;

The following two are *optional* if you only want to generate token ids from token
ids but compulsory to generate to or from text:

- `ids_to_text` which returns text from token ids;
- `text_to_ids` which returns text from token ids;

If you do not want to modify your model, you can simply define a simple wrapper class:

```python
from txt import Model

class MyModel(Model):
  self.decoder = MyModel
  self.tokenizer = MyTokenizer

  def decode(self, token_ids):
    return MyModel(token_ids)

  def text_to_ids(self, text):
    return self.tokenizer.encode(text)

  def ids_to_text(self, token_ids):
    return self.tokenizer.decode(token_ids)
```

### 
