"""Launch the inference server for Llava-video model."""

import argparse

from sglang.srt.server import ServerArgs, launch_server

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)


    model_override_args = {}
    model_override_args["mm_spatial_pool_stride"] = 2
    model_override_args["architectures"] = ["AuroraCapForCausalLM"]
    model_override_args["num_frames"] = 16
    model_override_args["tome_ratio"] = 0.8
    model_override_args["model_type"] = "auroracap"

    launch_server(server_args, model_override_args, None)
