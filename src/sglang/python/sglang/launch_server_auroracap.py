"""Launch the inference server for AuroraCap model."""

import json
import sys

from sglang.srt.server import launch_server
from sglang.srt.server_args import prepare_server_args

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])


    model_override_args = {}
    model_override_args["mm_spatial_pool_stride"] = 2
    model_override_args["architectures"] = ["AuroraCapForCausalLM"]
    model_override_args["num_frames"] = 16
    model_override_args["tome_ratio"] = 0.8
    model_override_args["model_type"] = "llava"

    server_args.json_model_override_args = json.dumps(model_override_args)
    launch_server(server_args)
