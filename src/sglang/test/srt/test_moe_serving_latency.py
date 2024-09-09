import os
import subprocess
import unittest

from sglang.srt.utils import kill_child_process
from sglang.test.test_utils import DEFAULT_MOE_MODEL_NAME_FOR_TEST


class TestServingLatency(unittest.TestCase):
    def test_default(self):
        command = [
            "python3",
            "-m",
            "sglang.bench_latency",
            "--model",
            DEFAULT_MOE_MODEL_NAME_FOR_TEST,
            "--batch-size",
            "1",
            "--input",
            "128",
            "--output",
            "8",
            "--tp",
            "2",
        ]
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        output = stdout.decode()
        error = stderr.decode()
        print(f"Output: {output}")
        print(f"Error: {error}")

        lastline = output.split("\n")[-3]
        value = float(lastline.split(" ")[-2])

        if os.getenv("SGLANG_IS_IN_CI", "false") == "true":
            assert value > 125

        kill_child_process(process.pid)


if __name__ == "__main__":
    unittest.main()
