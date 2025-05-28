from evals.base import Evaluator


class ReLUEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("ReLU", eval_cfg, **kwargs)
