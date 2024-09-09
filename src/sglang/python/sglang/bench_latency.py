"""
Benchmark the latency of a given model. It accepts arguments similar to those of launch_server.py.

# Usage (latency test)
## with dummy weights:
python -m sglang.bench_latency --model-path meta-llama/Meta-Llama-3-8B-Instruct --load-format dummy
## sweep through multiple data points and store (append) the results in a jsonl file:
python -m sglang.bench_latency --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 12 14 --input-len 256 512 --output-len 32 256 --result-filename out.jsonl
## do some changes, and store the results under a different run_name:
python -m sglang.bench_latency --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch 1 12 14 --input-len 256 512 --output-len 32 256 --result-filename out.jsonl --run-name after
## plot the results in series of lines:
python -m sglang.bench_latency --result-filename out.jsonl --graph-sql="select run_name, batch_size, prefill_throughput from results"

# Usage (correctness test):
python -m sglang.bench_latency --model-path TinyLlama/TinyLlama-1.1B-Chat-v0.4 --correct

## Reference output (of the correctness test above, can be gpu dependent):
input_ids=[[1, 450, 7483, 310, 3444, 338], [1, 450, 7483, 310, 278, 3303, 13187, 290, 338], [1, 20628, 338, 263, 6575, 1460, 2462, 322, 306, 763]]

prefill logits (first half): tensor([[-10.0312,  -9.5000,   0.8931,  ...,  -4.9414,  -3.2422,  -3.3633],
        [-10.0312,  -9.5000,   0.8931,  ...,  -4.9414,  -3.2422,  -3.3633],
        [ -9.1875, -10.2500,   2.7129,  ...,  -4.3359,  -4.0664,  -4.1328]],
       device='cuda:0')

prefill logits (final): tensor([[-8.3125, -7.1172,  3.3457,  ..., -4.9570, -4.1328, -3.4141],
        [-8.9141, -9.0156,  4.1445,  ..., -4.9922, -4.4961, -4.0781],
        [-9.6328, -9.0547,  4.0195,  ..., -5.3047, -4.7148, -4.4570]],
       device='cuda:0')

========== Prompt 0 ==========
<s> The capital of France is Paris.
The capital of the United States is Washington, D.C.


========== Prompt 1 ==========
<s> The capital of the United Kindom is London.
The capital of the United Kingdom is London.
The capital of the

========== Prompt 2 ==========
<s> Today is a sunny day and I like to go for a walk in the park.
I'm going to the park
"""

import argparse
import dataclasses
import itertools
import logging
import multiprocessing
import os
import sqlite3
import time
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_config import ModelConfig
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import suppress_other_loggers


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "before"
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (1024,)
    output_len: Tuple[int] = (16,)
    result_filename: str = ""
    correctness_test: bool = False
    # This is only used for correctness test
    cut_len: int = 4
    # Plotting args
    graph_sql: str = (
        "select run_name, batch_size, prefill_throughput from results where run_name='before'"
    )
    graph_filename: str = "out.png"

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, nargs="+", default=BenchArgs.batch_size
        )
        parser.add_argument(
            "--input-len", type=int, nargs="+", default=BenchArgs.input_len
        )
        parser.add_argument(
            "--output-len", type=int, nargs="+", default=BenchArgs.output_len
        )
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument("--correctness-test", action="store_true")
        parser.add_argument("--cut-len", type=int, default=BenchArgs.cut_len)
        # graphing
        parser.add_argument("--graph-sql", type=str, default=BenchArgs.graph_sql)
        parser.add_argument(
            "--graph-filename", type=str, default=BenchArgs.graph_filename
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to case the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: attr_type(getattr(args, attr)) for attr, attr_type in attrs}
        )


def load_model(server_args, tp_rank):
    suppress_other_loggers()
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    model_config = ModelConfig(
        server_args.model_path,
        server_args.trust_remote_code,
        context_length=server_args.context_length,
    )
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=tp_rank,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        nccl_port=28888,
        server_args=server_args,
    )
    rank_print(f"max_total_num_tokens={model_runner.max_total_num_tokens}")
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    if server_args.tp_size > 1:
        dist.barrier()
    return model_runner, tokenizer


def prepare_inputs_for_correctness_test(bench_args, tokenizer):
    prompts = [
        "The capital of France is",
        "The capital of the United Kindom is",
        "Today is a sunny day and I like",
    ]
    input_ids = [tokenizer.encode(p) for p in prompts]
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(prompts)):
        assert len(input_ids[i]) > bench_args.cut_len

        tmp_input_ids = input_ids[i][: bench_args.cut_len]
        req = Req(rid=i, origin_input_text=prompts[i], origin_input_ids=tmp_input_ids)
        req.prefix_indices = []
        req.sampling_params = sampling_params
        req.fill_ids = req.origin_input_ids
        reqs.append(req)

    return input_ids, reqs


def prepare_extend_inputs_for_correctness_test(
    bench_args, input_ids, reqs, model_runner
):
    for i in range(len(reqs)):
        req = reqs[i]
        req.fill_ids += input_ids[i][bench_args.cut_len :]
        req.prefix_indices = model_runner.req_to_token_pool.req_to_token[
            i, : bench_args.cut_len
        ]
    return reqs


def prepare_synthetic_inputs_for_latency_test(batch_size, input_len):
    input_ids = np.ones((batch_size, input_len), dtype=np.int32)
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(input_ids)):
        req = Req(rid=i, origin_input_text="", origin_input_ids=list(input_ids[i]))
        req.prefix_indices = []
        req.sampling_params = sampling_params
        req.fill_ids = req.origin_input_ids
        reqs.append(req)

    return reqs


def extend(reqs, model_runner):
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool=model_runner.token_to_kv_pool,
        tree_cache=None,
    )
    batch.prepare_for_extend(model_runner.model_config.vocab_size)
    sample_output, logits_output = model_runner.forward(batch, ForwardMode.EXTEND)
    next_token_ids = sample_output.batch_next_token_ids.tolist()
    return next_token_ids, logits_output.next_token_logits, batch


def decode(input_token_ids, batch, model_runner):
    batch.prepare_for_decode(input_token_ids)
    sample_output, logits_output = model_runner.forward(batch, ForwardMode.DECODE)
    next_token_ids = sample_output.batch_next_token_ids.tolist()
    return next_token_ids, logits_output.next_token_logits


@torch.inference_mode()
def correctness_test(
    server_args,
    bench_args,
    tp_rank,
):
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer = load_model(server_args, tp_rank)

    # Prepare inputs
    input_ids, reqs = prepare_inputs_for_correctness_test(bench_args, tokenizer)
    rank_print(f"\n{input_ids=}\n")

    if bench_args.cut_len > 0:
        # Prefill
        next_token_ids, next_token_logits, batch = extend(reqs, model_runner)
        rank_print(f"prefill logits (first half): {next_token_logits} \n")

    # Prepare extend inputs
    reqs = prepare_extend_inputs_for_correctness_test(
        bench_args, input_ids, reqs, model_runner
    )

    # Extend
    next_token_ids, next_token_logits, batch = extend(reqs, model_runner)
    rank_print(f"prefill logits (final): {next_token_logits} \n")

    # Decode
    output_ids = [input_ids[i] + [next_token_ids[i]] for i in range(len(input_ids))]
    for _ in range(bench_args.output_len[0]):
        next_token_ids, _ = decode(next_token_ids, batch, model_runner)
        for i in range(len(reqs)):
            output_ids[i].append(next_token_ids[i])

    # Print
    for i in range(len(reqs)):
        rank_print(f"========== Prompt {i} ==========")
        rank_print(tokenizer.decode(output_ids[i]), "\n")


@torch.inference_mode()
def latency_test_run_once(
    run_name, model_runner, rank_print, reqs, batch_size, input_len, output_len
):
    max_batch_size = model_runner.max_total_num_tokens // (input_len + output_len)
    if batch_size > max_batch_size:
        rank_print(
            f"skipping ({batch_size}, {input_len}, {output_len}) due to max batch size limit"
        )
        return

    # Clear the pools.
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool.clear()

    measurement_results = {
        "run_name": run_name,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
    }

    tot_latency = 0

    # Prefill
    torch.cuda.synchronize()
    tic = time.time()
    next_token_ids, _, batch = extend(reqs, model_runner)
    torch.cuda.synchronize()
    prefill_latency = time.time() - tic
    tot_latency += prefill_latency
    throughput = input_len * batch_size / prefill_latency
    rank_print(
        f"Prefill. latency: {prefill_latency:6.5f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["prefill_latency"] = prefill_latency
    measurement_results["prefill_throughput"] = throughput

    # Decode
    decode_latencies = []
    for i in range(output_len):
        torch.cuda.synchronize()
        tic = time.time()
        next_token_ids, _ = decode(next_token_ids, batch, model_runner)
        torch.cuda.synchronize()
        latency = time.time() - tic
        tot_latency += latency
        throughput = batch_size / latency
        decode_latencies.append(latency)
        if i < 5:
            rank_print(
                f"Decode.  latency: {latency:6.5f} s, throughput: {throughput:9.2f} token/s"
            )
    med_decode_latency = np.median(decode_latencies)
    med_decode_throughput = batch_size / med_decode_latency
    rank_print(
        f"Decode.  median latency: {med_decode_latency:6.5f} s, median throughput: {med_decode_throughput:9.2f} token/s"
    )
    measurement_results["median_decode_latency"] = med_decode_latency
    measurement_results["median_decode_throughput"] = med_decode_throughput

    throughput = (input_len + output_len) * batch_size / tot_latency
    rank_print(
        f"Total. latency: {tot_latency:6.3f} s, throughput: {throughput:9.2f} token/s"
    )
    measurement_results["total_latency"] = tot_latency
    measurement_results["total_throughput"] = throughput
    return measurement_results


def latency_test(
    server_args,
    bench_args,
    tp_rank,
):
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    # Load the model
    model_runner, tokenizer = load_model(server_args, tp_rank)

    # Prepare inputs for warm up
    reqs = prepare_synthetic_inputs_for_latency_test(
        bench_args.batch_size[0], bench_args.input_len[0]
    )

    # Warm up
    rank_print("Warmup ...")
    latency_test_run_once(
        bench_args.run_name,
        model_runner,
        rank_print,
        reqs,
        bench_args.batch_size[0],
        bench_args.input_len[0],
        4,  # shorter decoding to speed up the warmup
    )
    rank_print("Benchmark ...")

    # Run the sweep
    result_list = []
    for bs, il, ol in itertools.product(
        bench_args.batch_size, bench_args.input_len, bench_args.output_len
    ):
        reqs = prepare_synthetic_inputs_for_latency_test(bs, il)
        ret = latency_test_run_once(
            bench_args.run_name, model_runner, rank_print, reqs, bs, il, ol
        )
        if ret is not None:
            result_list.append(ret)

    # Write results in jsonlines format on rank 0.
    if tp_rank == 0 and bench_args.result_filename:
        import jsonlines

        with jsonlines.open(bench_args.result_filename, "a") as f:
            f.write_all(result_list)


def plot_latency_test(
    server_args,
    bench_args,
    tp_rank,
):
    assert tp_rank == 0

    # read the jsonl file and put in sqlite
    df = pd.read_json(bench_args.result_filename, lines=True)
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()

    # get the columns and their types
    column_names = list(df.iloc[0].keys())
    type_dict = {
        str: "TEXT",
        np.int64: "INTEGER",
        np.float64: "FLOAT",
    }
    column_types = [type_dict[type(i)] for i in list(df.iloc[0])]

    # create the table
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS results (
            {", ".join([f"{name} {type}" for name, type in zip(column_names, column_types)])}
        )
    """
    )
    conn.commit()

    # write the results to DB
    df.to_sql("results", conn, if_exists="replace", index=False)
    conn.commit()

    # read it back using sql
    df = pd.read_sql_query(bench_args.graph_sql, conn)
    conn.close()

    # plot it and save to a file
    import matplotlib.pyplot as plt

    assert (
        len(df.columns) == 3
    ), f"The sql should have fetched <series, x, y> columns, not {df.columns}"
    for label in df[df.columns[0]].unique():
        q = f"{df.columns[0]}=='{label}'"
        series = df.query(q)
        plt.plot(series[df.columns[1]], series[df.columns[2]], label=q, marker="o")
    plt.xlabel(df.columns[1])
    plt.ylabel(df.columns[2])
    plt.legend()
    plt.savefig(bench_args.graph_filename, dpi=300)

    # if in kitty, just dump it to the terminal
    if os.environ["TERM"] == "xterm-kitty":
        os.system(
            f"kitty icat --use-window-size 1,1,600,600 {bench_args.graph_filename}"
        )


def main(server_args, bench_args):

    if server_args.model_path:
        if bench_args.correctness_test:
            work_func = correctness_test
        else:
            work_func = latency_test
    elif os.path.isfile(bench_args.result_filename):
        assert bench_args.graph_filename, "please provide a filename for the graph"
        work_func = plot_latency_test
    else:
        raise ValueError(
            "Provide --model-path for running the tests or "
            "provide --result-filename for plotting the results"
        )

    if server_args.tp_size == 1:
        work_func(server_args, bench_args, 0)
    else:
        workers = []
        for tp_rank in range(server_args.tp_size):
            proc = multiprocessing.Process(
                target=work_func,
                args=(
                    server_args,
                    bench_args,
                    tp_rank,
                ),
            )
            proc.start()
            workers.append(proc)

        for proc in workers:
            proc.join()

        proc.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    # For this script, model-path is not required
    assert (
        parser._actions[1].option_strings[0] == "--model-path"
    ), "options changed, this code need to be updated"
    parser._actions[1].required = False
    args = parser.parse_args()

    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    main(server_args, bench_args)
