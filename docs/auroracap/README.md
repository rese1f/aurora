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
    --model_path wchai/AuroraCap-7B-VID-xtuner \
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

## FAQ

Q: Can I only use token merging during inference?

A: No, our experiments show that token merging is also a way to accelerate training while maintaining similar performance. Additionally, besides auroracap, you can also use token merging on other llava-like models.

Q: How should I set the `token_kept_ratio` parameter?

A: AuroraCap uses token merging technique to reduce the number of visual tokens before fed into the llm decoder. We using `token_kept_ratio` range from 0 to 1 to control the number of visual tokens kept. For example, if `token_kept_ratio` is 0.5, then 50% of the visual tokens will be kept. We recommend to use `token_kept_ratio` in the range of 0.2 to 0.4 for better performance-cost trade-off for captioning tasks, above 0.5 for visual question answering tasks, and above 0.8 for OCR-related tasks.

Q: Why do we provide both Huggingface format and Xtuner format weights for AuroraCap?

A: While Xtuner supports saving checkpoints in multiple formats, it currently only allows continued training with the Xtuner format. Therefore, we currently provide the model in the Xtuner format for both continued training and inference. In the future, we will provide the model in the Huggingface format for both training and inference, enabling quicker SGLang deployment and integration with the transformers.

## Citation
