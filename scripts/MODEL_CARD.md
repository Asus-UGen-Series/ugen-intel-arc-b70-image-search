---
license: apache-2.0
library_name: openvino
tags:
  - openvino
  - siglip2
  - image-text-retrieval
  - naflex
base_model: google/siglip2-so400m-patch16-naflex
---

# SigLIP 2 SO400M NaFlex — OpenVINO FP16 for Visual Search

This package is an OpenVINO FP16 conversion of
[`google/siglip2-so400m-patch16-naflex`](https://huggingface.co/google/siglip2-so400m-patch16-naflex)
for local image-text retrieval in Visual Search.

The expensive transformer uses a fixed 256-patch input for stable Intel GPU
throughput. Aspect-ratio-dependent position embeddings are an explicit input,
so portrait phone photos and wide screenshots keep NaFlex behavior without
triggering a new GPU graph for every image shape.

## Files and inputs

- `vision_model.xml/.bin`: inputs `pixel_values [1,256,768]`,
  `pixel_attention_mask [1,256]`, `position_embeddings [1,256,1152]`.
- `text_model.xml/.bin`: multilingual text encoder.
- `position_embedding.npy`: the trained 2-D base position grid.
- tokenizer and processor configuration copied from the pinned upstream model.

## License and attribution

The upstream model is Apache-2.0. See its model card for training data, intended
uses, limitations, and the SigLIP 2 paper citation. This repository contains a
format conversion, not a newly trained model.
