# NCM Interactive Web Book

An interactive web book that teaches **Next-Concept Modeling (NCM)** — fast, parallel, coarse-to-fine text generation — starting from absolute zero (no prior knowledge of language models assumed) and ending at every equation of the research paper.

**Read it:** open `index.html` in a browser, or visit the GitHub Pages site for this repository.

## What's inside

- 26 chapters in four parts: Foundations → The NCM tokenizer → The NCM generator → Evidence, plus an equation cheat sheet and glossary.
- ~45 interactive widgets: softmax and sampling, embeddings, attention masks, vector quantization with live k-means, residual and multiscale residual quantization, a working toy NCM tokenizer and coarse-to-fine sampler that run in the browser, loss weighting, conditioning corruption, classifier-free guidance, MBR decoding, a cost model, the ELBO, metrics (perplexity, entropy, BLEU, ROUGE), a results explorer for the paper's tables, and a replay of the paper's Sudoku solving trajectory.
- 50 self-check quiz questions.

## Sources

- Paper: Kumbong, Khare, Mlauzi, Woen, Ding, Buchanan, Arora, Mirhoseini, Ré. *Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation*. [arXiv:2610.14162](https://arxiv.org/abs/2610.14162), 2026.
- Code: [github.com/HazyResearch/ncm](https://github.com/HazyResearch/ncm) (MIT). Figures in `assets/img/` marked as coming from that repository are used under its MIT license (`assets/img/LICENSE-NCM-repo.txt`).

This is an independent educational companion, not an official publication of the paper's authors. Numbers quoted from the paper come from arXiv v1; toy-model numbers come from the small model that runs in the page.

## Structure

```
index.html                 the book
assets/css/book.css        styles (light and dark)
assets/js/core.js          navigation, progress, quizzes, chart helpers
assets/js/toyncm.js        the in-browser toy multiscale residual VQ + sampler
assets/js/widgets-*.js     interactive widgets per part
assets/img/                figures from the NCM repository (MIT)
```

No build step; it is a static site. Math is rendered with KaTeX from a CDN.
