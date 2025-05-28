import numpy as np
import torch
from torch.utils.data import DataLoader

from evals.metrics.base import unlearning_metric
from evals.metrics.utils import run_batchwise_evals

from evals.metrics.utils import evaluate_correct_sequence_probability

@unlearning_metric(name="correct_sequence_probability")
def correct_sequence_probability(model, **kwargs):
    """Calculate the probability of the correct sequence in the dataset."""
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    
    assert batch_size == 1, ValueError("Batch size must be 1 for this metric.")
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collator)

    fun_args = {}
    scores_by_index = run_batchwise_evals(
        model, dataloader, evaluate_correct_sequence_probability, fun_args, "Calculating correct sequence probability"
    )
    prob_values = np.array([evals["normalized_sequence_probability"] for evals in scores_by_index.values()])
    prob_values = prob_values.squeeze()
    return {"agg_value": np.mean(prob_values), "value_by_index": scores_by_index}
    