from .chat import ChatTemplate
from .hybrid import HybridChatTemplate

CHAT_TEMPLATE_MAP = {
    'internlm2':
    HybridChatTemplate(
        system='<|im_start|>system\n{system}<|im_end|>\n',
        user='<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n',
        assistant='{assistant}<|im_end|>',
        stop_words=['<|im_end|>']
    ),
    'vicuna':
    HybridChatTemplate(
        system='A chat between a curious user and an artificial '
                'intelligence assistant. The assistant gives '
                'helpful, detailed, and polite answers to the '
                'user\'s questions. {system}\n ',
        user='USER: {input} ASSISTANT:',
        # TODO: check with assistant and stop_words
        assistant='{assistant}</s>',
        stop_words=['</s>']
    ),
}

__all__ = ['ChatTemplate', 'HybridChatTemplate']
