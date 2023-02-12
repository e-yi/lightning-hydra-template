from typing import Dict, Iterable, List, Union

import torch.nn
from pytorch_lightning.utilities.memory import recursive_detach
from torch.nn import Module

from src.evaluations import LEVEL_STAGE, LEVEL_EPOCH, LEVEL_BATCH, PHASE_TEST, PHASE_TRAIN, PHASE_VALID
from src.evaluations.metrics.base_metrics import BaseMetaMetric, MyMetric, BaseMetric


class MetricGroup(Module):
    """

    name:
    phase: PHASE_VALID, PHASE_TRAIN, PHASE_TEST
    """

    def __init__(self, input_metrics: List[MyMetric]):
        super().__init__()

        metrics = []
        meta_metrics = []
        for m in input_metrics:
            if isinstance(m, BaseMetric):
                metrics.append(m)
            elif isinstance(m, BaseMetaMetric):
                meta_metrics.append(m)
            else:
                raise ValueError('?')

        self.metrics = torch.nn.ModuleList(metrics)
        self.meta_metrics = torch.nn.ModuleList(meta_metrics)

    def __update_metrics(self, val: Dict, phase: str):
        for metric in self.metrics:
            metric.batch_update(**val, phase=phase)

    def __update_meta_metrics(self, metrics_val: Dict, phase: str):
        for meta_metric in self.meta_metrics:
            meta_metric.epoch_update(**metrics_val, phase=phase)

    def batch_step(self, val: Dict, phase: str):
        val = recursive_detach(val)

        self.__update_metrics(val, phase)

        metrics = {}
        for metric in self.metrics:
            metrics.update(metric.get_value(phase=phase, level=LEVEL_BATCH))

        return {f'{phase}/{k}': v for k, v in metrics.items()}

    def epoch_step(self, phase: str):
        metrics = {}
        for metric in self.metrics:
            metrics.update(metric.get_value(phase=phase, level=LEVEL_EPOCH))

        self.__update_meta_metrics(metrics, phase)

        for meta_metric in self.meta_metrics:
            metrics.update(meta_metric.get_value(phase=phase))

        return {f'{phase}/{k}': v for k, v in metrics.items()}

    def reset(self, level, phases: Union[str, Iterable[str]] = (PHASE_TRAIN, PHASE_VALID, PHASE_TEST)):
        """

        :param level: LEVEL_EPOCH or LEVEL_STAGE
        :param phases:
        :return:
        """
        if level == LEVEL_BATCH:
            raise ValueError(f'wrong reset level {level}')

        if isinstance(phases, str):
            phases = (phases,)

        for metric in self.metrics:
            for phase_i in phases:
                metric.reset(phase_i)

        if level == LEVEL_STAGE:
            for meta_metric in self.meta_metrics:
                for phase_i in phases:
                    meta_metric.reset(phase_i)
