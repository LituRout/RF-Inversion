<div align="center">
<h1>Semantic Image Inversion and Editing using Rectified Stochastic Differential Equations</h1>

<a href='https://rf-inversion.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/pdf/2410.10792'><img src='https://img.shields.io/badge/ArXiv-Preprint-red'></a>
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Demo-blue)](https://github.com/logtd/ComfyUI-Fluxtapoz)
[![GitHub](https://img.shields.io/github/stars/LituRout/RF-Inversion?style=social)](https://github.com/LituRout/RF-Inversion)
</div>


Rectified flows for image inversion and editing. Our approach efficiently inverts reference style images in (a) and (b) without requiring text descriptions of the images and applies desired edits based on new prompts (e.g. “a girl” or “a dwarf”). For a reference content image (e.g. a cat in (c) or a face in (d)), it performs semantic image editing  e.g. “ sleeping cat”) and stylization (e.g. “a photo of a cat in origmai style”) based on prompts, without leaking unwanted content from the reference image (input images have orange borders).

![teaser](./data/main.png)


## 🔥 Updates
- **[2024.10.15]** [Code](https://github.com/logtd/ComfyUI-Fluxtapoz) reimplemented by open-source ComfyUI community, thanks [logtd](https://github.com/logtd)!
- **[2024.10.14]** [Paper](https://arxiv.org/pdf/2410.10792) is published on arXiv!


## 🚀 Comfy User Interface
Try ComfyUI <a href='https://github.com/comfyanonymous/ComfyUI'><img src='https://img.shields.io/github/stars/comfyanonymous/ComfyUI'></a> for better experience:
[ComfyUI Node](https://github.com/logtd/ComfyUI-Fluxtapoz)🔥

```
# Install [ComfyUI](https://comfyanonymous.github.io/ComfyUI_examples/flux/) to run flux
1. > cd ComfyUI
2. > python main.py

# Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager). 
1. > cd ComfyUI/custom_nodes
2. > git clone https://github.com/ltdrdata/ComfyUI-Manager.git
3. > cd ..
4. > python main.py

# Install [RF-Inversion](https://rf-inversion.github.io/) ComfyUI [Node](https://github.com/logtd/ComfyUI-Fluxtapoz)
1. Click on "Manager"
2. Install via Git URL: [https://github.com/logtd/ComfyUI-Fluxtapoz](https://github.com/logtd/ComfyUI-Fluxtapoz)
3. If you see error, change security level in ComfyUI/custom_nodes/ComfyUI-Manager/config.ini from "normal" to "weak"
4. > cd ComfyUI
5. > python main.py
6. Copy RF-Inversion [workflow](https://github.com/logtd/ComfyUI-Fluxtapoz/blob/main/example_workflows/example_rf_inversion_updated.json) and paste on the ComfyUI window.
7. Click on "Queue Prompt" to see the result
8. Tune hyper-parameters (such as eta, start_step, stop_step) to get the desired outcome.
```
## Citation

```
@article{rout2024rfinversion,
  title={Semantic Image Inversion and Editing using Rectified Stochastic Differential Equations},
  author={Litu Rout and Yujia Chen and Nataniel Ruiz and Constantine Caramanis and Sanjay Shakkottai and Wen-Sheng Chu},
  journal={arXiv preprint arXiv:2410.10792},
  year={2024}
}
```

<!-- ## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=LituRout/RB-Modulation&type=Date)](https://star-history.com/#LituRout/RB-Modulation&Date) -->

## Licenses

Copyright © 2024, Google LLC. All rights reserved.
