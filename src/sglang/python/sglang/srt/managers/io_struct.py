"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The definition of objects transfered between different
processes (TokenizerManager, DetokenizerManager, Controller).
"""

import copy
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from sglang.srt.managers.schedule_batch import BaseFinishReason
from sglang.srt.sampling.sampling_params import SamplingParams


@dataclass
class GenerateReqInput:
    # The input prompt. It can be a single prompt or a batch of prompts.
    text: Optional[Union[List[str], str]] = None
    # The token ids for text; one can either specify text or input_ids.
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    # The image input. It can be a file name, a url, or base64 encoded string.
    # See also python/sglang/srt/utils.py:load_image.
    image_data: Optional[Union[List[str], str]] = None
    # The sampling_params. See descriptions below.
    sampling_params: Union[List[Dict], Dict] = None
    # The request id.
    rid: Optional[Union[List[str], str]] = None
    # Whether to return logprobs.
    return_logprob: Optional[Union[List[bool], bool]] = None
    # If return logprobs, the start location in the prompt for returning logprobs.
    logprob_start_len: Optional[Union[List[int], int]] = None
    # If return logprobs, the number of top logprobs to return at each position.
    top_logprobs_num: Optional[Union[List[int], int]] = None
    # Whether to detokenize tokens in text in the returned logprobs.
    return_text_in_logprobs: bool = False
    # Whether to stream output.
    stream: bool = False

    def post_init(self):
        if (self.text is None and self.input_ids is None) or (
            self.text is not None and self.input_ids is not None
        ):
            raise ValueError("Either text or input_ids should be provided.")

        if (
            isinstance(self.sampling_params, dict)
            and self.sampling_params.get("n", 1) != 1
        ):
            is_single = False
        else:
            if self.text is not None:
                is_single = isinstance(self.text, str)
            else:
                is_single = isinstance(self.input_ids[0], int)
        self.is_single = is_single

        if is_single:
            if self.sampling_params is None:
                self.sampling_params = {}
            if self.rid is None:
                self.rid = uuid.uuid4().hex
            if self.return_logprob is None:
                self.return_logprob = False
            if self.logprob_start_len is None:
                self.logprob_start_len = -1
            if self.top_logprobs_num is None:
                self.top_logprobs_num = 0
        else:
            parallel_sample_num_list = []
            if isinstance(self.sampling_params, dict):
                parallel_sample_num = self.sampling_params.get("n", 1)
            elif isinstance(self.sampling_params, list):
                for sp in self.sampling_params:
                    parallel_sample_num = sp.get("n", 1)
                    parallel_sample_num_list.append(parallel_sample_num)
                parallel_sample_num = max(parallel_sample_num_list)
                all_equal = all(
                    element == parallel_sample_num
                    for element in parallel_sample_num_list
                )
                if parallel_sample_num > 1 and (not all_equal):
                    # TODO cope with the case that the parallel_sample_num is different for different samples
                    raise ValueError(
                        "The parallel_sample_num should be the same for all samples in sample params."
                    )
            else:
                parallel_sample_num = 1
            self.parallel_sample_num = parallel_sample_num

            if parallel_sample_num != 1:
                # parallel sampling +1 represents the original prefill stage
                num = parallel_sample_num + 1
                if isinstance(self.text, list):
                    # suppot batch operation
                    self.batch_size = len(self.text)
                    num = num * len(self.text)
                elif isinstance(self.input_ids, list) and isinstance(
                    self.input_ids[0], list
                ):
                    self.batch_size = len(self.input_ids)
                    num = num * len(self.input_ids)
                else:
                    self.batch_size = 1
            else:
                # support select operation
                num = len(self.text) if self.text is not None else len(self.input_ids)
                self.batch_size = num

            if self.image_data is None:
                self.image_data = [None] * num
            elif not isinstance(self.image_data, list):
                self.image_data = [self.image_data] * num

            if self.sampling_params is None:
                self.sampling_params = [{}] * num
            elif not isinstance(self.sampling_params, list):
                self.sampling_params = [self.sampling_params] * num

            if self.rid is None:
                self.rid = [uuid.uuid4().hex for _ in range(num)]
            else:
                if not isinstance(self.rid, list):
                    raise ValueError("The rid should be a list.")

            if self.return_logprob is None:
                self.return_logprob = [False] * num
            elif not isinstance(self.return_logprob, list):
                self.return_logprob = [self.return_logprob] * num

            if self.logprob_start_len is None:
                self.logprob_start_len = [-1] * num
            elif not isinstance(self.logprob_start_len, list):
                self.logprob_start_len = [self.logprob_start_len] * num

            if self.top_logprobs_num is None:
                self.top_logprobs_num = [0] * num
            elif not isinstance(self.top_logprobs_num, list):
                self.top_logprobs_num = [self.top_logprobs_num] * num


@dataclass
class TokenizedGenerateReqInput:
    # The request id
    rid: str
    # The input text
    input_text: str
    # The input token ids
    input_ids: List[int]
    # The pixel values for input images
    pixel_values: List[float]
    # The hash values of input images
    image_hashes: List[int]
    # The image sizes
    image_sizes: List[List[int]]
    # The sampling parameters
    sampling_params: SamplingParams
    # Whether to return the logprobs
    return_logprob: bool
    # If return logprobs, the start location in the prompt for returning logprobs.
    logprob_start_len: int
    # If return logprobs, the number of top logprobs to return at each position.
    top_logprobs_num: int
    # Whether to stream output
    stream: bool


@dataclass
class EmbeddingReqInput:
    # The input prompt. It can be a single prompt or a batch of prompts.
    text: Optional[Union[List[str], str]] = None
    # The token ids for text; one can either specify text or input_ids.
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    # The request id.
    rid: Optional[Union[List[str], str]] = None
    # Dummy sampling params for compatibility
    sampling_params: Union[List[Dict], Dict] = None

    def post_init(self):
        if (self.text is None and self.input_ids is None) or (
            self.text is not None and self.input_ids is not None
        ):
            raise ValueError("Either text or input_ids should be provided.")

        if self.text is not None:
            is_single = isinstance(self.text, str)
        else:
            is_single = isinstance(self.input_ids[0], int)
        self.is_single = is_single

        if is_single:
            if self.rid is None:
                self.rid = uuid.uuid4().hex
            if self.sampling_params is None:
                self.sampling_params = {}
            self.sampling_params["max_new_tokens"] = 1
        else:
            # support select operation
            self.batch_size = (
                len(self.text) if self.text is not None else len(self.input_ids)
            )
            if self.rid is None:
                self.rid = [uuid.uuid4().hex for _ in range(self.batch_size)]
            else:
                if not isinstance(self.rid, list):
                    raise ValueError("The rid should be a list.")
            if self.sampling_params is None:
                self.sampling_params = [{}] * self.batch_size
            for i in range(self.batch_size):
                self.sampling_params[i]["max_new_tokens"] = 1


@dataclass
class TokenizedEmbeddingReqInput:
    # The request id
    rid: str
    # The input text
    input_text: str
    # The input token ids
    input_ids: List[int]
    # Dummy sampling params for compatibility
    sampling_params: SamplingParams


@dataclass
class BatchTokenIDOut:
    # The request id
    rids: List[str]
    # The version id to sync decode status with in detokenizer_manager
    vids: List[int]
    decoded_texts: List[str]
    decode_ids: List[int]
    read_offsets: List[int]
    skip_special_tokens: List[bool]
    spaces_between_special_tokens: List[bool]
    meta_info: List[Dict]
    finished_reason: List[BaseFinishReason]

    def __post_init__(self):
        # deepcopy meta_info to avoid modification in place
        self.meta_info = copy.deepcopy(self.meta_info)


@dataclass
class BatchStrOut:
    # The request id
    rids: List[str]
    # The output decoded strings
    output_strs: List[str]
    # The meta info
    meta_info: List[Dict]
    # The finish reason
    finished_reason: List[BaseFinishReason]


@dataclass
class BatchEmbeddingOut:
    # The request id
    rids: List[str]
    # The output embedding
    embeddings: List[List[float]]
    # The meta info
    meta_info: List[Dict]
    # The finish reason
    finished_reason: List[BaseFinishReason]


@dataclass
class FlushCacheReq:
    pass


@dataclass
class UpdateWeightReqInput:
    # The model path with the new weights
    model_path: str
    # The format to load the weights
    load_format: Optional[str] = None


@dataclass
class UpdateWeightReqOutput:
    success: bool
    message: str


@dataclass
class AbortReq:
    # The request id
    rid: str
