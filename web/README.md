---
title: Hedwig Storyteller
emoji: 🦉
colorFrom: indigo
colorTo: yellow
sdk: static
app_file: index.html
license: mit
short_description: A tiny storytelling LLM that runs in your browser
---

# 🦉 Hedwig: a storytelling language model built from scratch

Type a prompt like *"Write a short story about Lily and a shiny key"* and watch Hedwig write it.
Click any word for the **🔬 Brain X-Ray**: the top 5 pieces Hedwig was choosing between.

Hedwig runs **inside your browser** in plain JavaScript (`hedwig.js`), with no server. The JavaScript brain
was checked against PyTorch and picks the same 40 greedy pieces out of 40.

| Brain | Knobs | Download (8-bit) |
|---|---|---|
| Mac Hedwig (default, fast) | 13.8M | 14 MB |
| Cloud Hedwig | 97.6M | 98 MB |

For scale: GPT-2 small is about 124M knobs and GPT-3 about 175 billion.

- Code (MIT): https://github.com/ansarbuilds/hedwig
- Weights: [hedwig-97m](https://huggingface.co/ansarbuilds/hedwig-97m) · [hedwig-300m](https://huggingface.co/ansarbuilds/hedwig-300m)
