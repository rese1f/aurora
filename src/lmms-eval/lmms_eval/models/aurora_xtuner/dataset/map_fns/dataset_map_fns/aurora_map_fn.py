# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.utils import DEFAULT_IMAGE_TOKEN


def aurora_image_only_map_fn(example):
    # input contains the DEFAULT_IMAGE_TOKEN only
    messages = example['conversations']
    input = ''
    conversation = []
    while messages and messages[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        messages = messages[1:]
    while messages and messages[-1]['from'] == 'human':
        # Skip the last one if it is from human
        messages = messages[:-1]
    for msg in messages:
        if msg['from'] == 'human':
            assert DEFAULT_IMAGE_TOKEN in msg['value']
            input += DEFAULT_IMAGE_TOKEN
        elif msg['from'] == 'gpt':
            conversation.append({'input': input, 'output': msg['value']})
            input = ''
        else:
            raise NotImplementedError
    return {'conversation': conversation}


def aurora_map_fn(example):
    messages = example['conversations']
    input = ''
    conversation = []
    while messages and messages[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        messages = messages[1:]
    while messages and messages[-1]['from'] == 'human':
        # Skip the last one if it is from human
        messages = messages[:-1]
    for msg in messages:
        if msg['from'] == 'human':
            if DEFAULT_IMAGE_TOKEN in msg['value']:
                img_num = msg['value'].count(DEFAULT_IMAGE_TOKEN)
                image_tokens = [DEFAULT_IMAGE_TOKEN] * img_num
                image_tokens = " ".join(image_tokens)
                msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                    '').strip()
                msg['value'] = image_tokens + '\n' + msg['value']
                msg['value'] = msg['value'].strip()
            input += msg['value']

        elif msg['from'] == 'gpt':
            conversation.append({'input': input, 'output': msg['value']})
            input = ''
        else:
            raise NotImplementedError
    
    return {'conversation': conversation}
