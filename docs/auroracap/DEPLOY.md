# Deploy
powered by [SGLang](https://github.com/sgl-project/sglang).

## Weight convert

For better deploy, we need to first convert model weight from .pth to official llava format.

```
python src/xtuner/xtuner/tools/model_converters/pth_to_hf.py \
    ${CONFIG_PATH} \
    ${PTH_PATH} \
    ${SAVE_PATH} \
    --save_format official
```

## SGLang server launch

To inference with SGLang, we need to launch the server.
```
python -m sglang.launch_server_auroracap \
    --model-path ${CHECKPOINT_PATH} \
    --port=30000 --chat-template=chatml-llava
```

We provide the example inference code for Aurora using another terminal.
```
from sglang import function, system, user, assistant, gen, set_default_backend, RuntimeEndpoint
import sglang as sgl

@sgl.function
def image_qa(s, image_file, question):
    s += sgl.user(sgl.image(image_file)+ question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=256))

set_default_backend(RuntimeEndpoint("http://localhost:30000"))

state = image_qa.run(
    image_file='input.png',
    question="Describe the image in detail"
)
print(state)
```