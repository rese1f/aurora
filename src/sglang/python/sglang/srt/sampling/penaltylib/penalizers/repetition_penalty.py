import typing

import torch

from ..orchestrator import _BatchedPenalizer, _TokenIDs


class BatchedRepetitionPenalizer(_BatchedPenalizer):
    """
    Repetition penalizer penalizes tokens based on their repetition in the input and output.
    """

    repetition_penalties: torch.Tensor = None
    cumulated_repetition_penalties: torch.Tensor = None

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.repetition_penalty != 1.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        self.cumulated_repetition_penalties = (
            torch.tensor(
                data=[1.0 for _ in self.orchestrator.reqs()],
                dtype=torch.float32,
                device=self.orchestrator.device,
            )
            .unsqueeze_(1)
            .repeat(1, self.orchestrator.vocab_size)
        )

        self.repetition_penalties = (
            torch.tensor(
                data=[
                    req.sampling_params.repetition_penalty
                    for req in self.orchestrator.reqs()
                ],
                dtype=torch.float32,
                device=self.orchestrator.device,
            )
            .unsqueeze_(1)
            .expand_as(self.cumulated_repetition_penalties)
        )

    def _teardown(self):
        del self.repetition_penalties
        del self.cumulated_repetition_penalties

        self.repetition_penalties = None
        self.cumulated_repetition_penalties = None

    def _cumulate_input_tokens(self, input_ids: _TokenIDs):
        mask = input_ids.occurrence_count() > 0
        self.cumulated_repetition_penalties[mask] = self.repetition_penalties[mask]

    def _cumulate_output_tokens(self, output_ids: _TokenIDs):
        mask = output_ids.occurrence_count() > 0
        self.cumulated_repetition_penalties[mask] = self.repetition_penalties[mask]

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.where(
            logits > 0,
            logits / self.cumulated_repetition_penalties,
            logits * self.cumulated_repetition_penalties,
        )

    def _filter(
        self, indices_to_keep: typing.List[int], indices_tensor_to_keep: torch.Tensor
    ):
        self.repetition_penalties = self.repetition_penalties[indices_tensor_to_keep]
        self.cumulated_repetition_penalties = self.cumulated_repetition_penalties[
            indices_tensor_to_keep
        ]

    def _merge(self, their: "BatchedRepetitionPenalizer"):
        self.repetition_penalties = torch.cat(
            [self.repetition_penalties, their.repetition_penalties], dim=0
        )
        self.cumulated_repetition_penalties = torch.cat(
            [self.cumulated_repetition_penalties, their.cumulated_repetition_penalties],
            dim=0,
        )
