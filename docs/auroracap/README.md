<img src="../../assets/auroracap/teaser.png" align="center">

## Resources

- [Website](https://rese1f.github.io/aurora-web/)
- [arXiv: Paper]()
- [GitHub: Code](https://github.com/rese1f/aurora)
- [Huggingface: AuroraCap Model](https://huggingface.co/collections/wchai/auroracap-66d117ffe13bedda96702013)
- [Huggingface: VDC Benchmark](https://huggingface.co/datasets/wchai/Video-Detailed-Caption)
- [Huggingface: Trainset](https://huggingface.co/datasets/wchai/AuroraCap-trainset)

## Docs

- [Train Docs](TRAIN.md) powered by [Xtuner](https://github.com/InternLM/xtuner).
- [Eval Docs](EVAL.md) powered by [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).
- [Deploy Docs](DEPLOY.md) powered by [SGLang](https://github.com/sgl-project/sglang).

[RETURN TO MAIN README](../../README.md)

## Features

AuroraCap is a efficient captioning model for image and video, achieving the best trade-off between performance and efficiency.

<img src="../../assets/auroracap/vdc_baseline.png" align="center">


## Quick Start

```
python inference.py \
    --model_path wchai/AuroraCap-7B-VID-hf \
    --prompt "Describe the video in detail." \
    --visual_input assets/auroracap/test.mp4 \
    --num_frm 8 \
    --token_kept_ratio 0.2 \
    --temperature 0.0 \
    --top_p 1.0 \
    --num_beams 1 \
    --max_new_tokens 2048
```

with Gradio GUI

```
python gradio_gui.py
```

## Citation
