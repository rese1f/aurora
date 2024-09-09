"""Global configurations"""


class GlobalConfig:
    def __init__(self):
        # Verbosity level
        # 0: do not output anything
        # 2: output final text after every run
        self.verbosity = 0

        # Default backend of the language
        self.default_backend = None

        # Runtime constants: New generation token ratio estimation
        self.init_new_token_ratio = 0.7
        self.base_min_new_token_ratio = 0.1
        self.new_token_ratio_decay = 0.001

        # Runtime constants: The threshold (number of tokens) to trigger layer-wise cuda sync.
        # This can improve the speed for large batch sizes during prefill.
        self.layer_sync_threshold = 8192

        # Runtime constants: others
        self.num_continue_decode_steps = 10
        self.retract_decode_steps = 20
        self.flashinfer_workspace_size = 384 * 1024 * 1024

        # Output tokenization configs
        self.skip_special_tokens_in_output = True
        self.spaces_between_special_tokens_in_out = True

        # Interpreter optimization configs
        self.eager_fill_image = False
        self.enable_precache_with_tracing = True
        self.enable_parallel_encoding = True
        self.enable_parallel_decoding = True

        # Deprecated
        # Choices: ["no_adjust", "adjust_cache"]
        # no_adjust: Do not adjust the position embedding of KV cache.
        # adjust_cache: Adjust the position embedding of KV cache.
        self.concate_and_append_mode = "no_adjust"


global_config = GlobalConfig()
