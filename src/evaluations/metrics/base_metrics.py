import abc
from abc import ABC
from typing import Dict, List

import torchmetrics
from torch import nn
from torch.nn import Module

PHASE_TRAIN = 'train'
PHASE_VALID = 'val'
PHASE_TEST = 'test'

PHASES = (PHASE_TRAIN, PHASE_VALID, PHASE_TEST)

LEVEL_BATCH = 'batch'
LEVEL_EPOCH = 'epoch'
LEVEL_STAGE = 'stage'

LEVELS = (LEVEL_BATCH, LEVEL_EPOCH, LEVEL_STAGE)


class MyMetric(ABC, Module):
    PHASE_PREFIX = 'phase_'
    pass


class BaseMetric(MyMetric):
    """
    updated each batch,
    input is the outputs of LightningModule.xxxx_step()
    log at each batch or(and) each epoch
    will be reset each epoch
    """

    def __init__(self, name: str, input_pos_args: List[str], phases: List[str], log_levels: List[str]) -> None:
        """

        :param name: metrics name, like `acc`. The name will be `<phase>/<name>_<log_level>` during logging.
        :param input_pos_args: keys of input data, **ordered**
        :param phases:
        :param log_levels:
        """
        super().__init__()
        self.name = name
        self.input_keys = input_pos_args
        self.phases = phases
        self.log_levels = log_levels

    @abc.abstractmethod
    def batch_update(self, phase, **kwargs):
        pass

    @abc.abstractmethod
    def get_value(self, phase, level) -> Dict:
        pass

    @abc.abstractmethod
    def reset(self, phase):
        pass


class BaseMetricAdapter(BaseMetric):

    def __init__(self, name: str, metric: torchmetrics.Metric,
                 input_pos_args: List[str], phases: List[str], log_levels: List[str],
                 input_kwargs: Dict = None) -> None:
        """

        :param name: metrics name, like `acc`. The name will be `<phase>/<name>_<log_level>` during logging.
        :param input_pos_args: keys of input data, **ordered**
        :param phases:
        :param log_levels:
        """
        super().__init__(name, input_pos_args, phases, log_levels)

        if input_kwargs is None:
            input_kwargs = {}
        self.input_kwargs = input_kwargs

        self.metrics = nn.ModuleDict({
            self.PHASE_PREFIX+p: metric() for p in phases
        })

        self.batch_val = {self.PHASE_PREFIX+i: None for i in phases}

    def batch_update(self, phase, **kwargs):
        if phase not in PHASES:
            raise ValueError(f'phase should be one of `{PHASES}`')

        if phase not in self.phases:
            return

        inputs = [kwargs[k] for k in self.input_keys]
        batch_val = self.metrics[self.PHASE_PREFIX+phase](*inputs, **self.input_kwargs)

        self.batch_val[self.PHASE_PREFIX+phase] = batch_val

    def get_value(self, phase, level) -> Dict:
        if phase not in PHASES:
            raise ValueError(f'phase should be one of `{PHASES}`')
        if level not in LEVELS:
            raise ValueError(f'level should be one of `{LEVELS}`')

        if phase not in self.phases:
            return {}

        if level == LEVEL_BATCH:
            val = self.batch_val[self.PHASE_PREFIX+phase]
        elif level == LEVEL_EPOCH:
            val = self.metrics[self.PHASE_PREFIX+phase].compute()
        else:
            raise ValueError(f'error level {level}')

        return {f'{self.name}_{level}': val}

    def reset(self, phase):
        if phase not in PHASES:
            raise ValueError(f'phase should be one of `{PHASES}`')

        if phase not in self.phases:
            return

        self.metrics[self.PHASE_PREFIX+phase].reset()


class BaseMetaMetric(MyMetric):
    """
    updated each epoch,
    input is other metrics(BaseMetric)
    log at each epoch
    will be reset once at the beginning
    """


    def __init__(self, name: str, input_pos_args: List[str], phases: List[str]) -> None:
        """

        :param name: metrics name, like `acc`. The name will be `<phase>/<name>` during logging.
        :param input_pos_args: input metrics' names,  **ordered**
        :param phases:
        """
        super().__init__()
        self.name = name
        self.input_keys = input_pos_args
        self.phases = phases

    @abc.abstractmethod
    def epoch_update(self, phase, **kwargs):
        pass

    @abc.abstractmethod
    def get_value(self, phase) -> Dict:
        pass

    @abc.abstractmethod
    def reset(self, phase):
        pass


class BaseMetaMetricAdapter(BaseMetaMetric):

    def __init__(self, name: str, metric: torchmetrics.Metric,
                 input_pos_args: List[str], phases: List[str],
                 input_kwargs: Dict = None) -> None:
        """

        :param name: metrics name, like `acc`. The name will be `<phase>/<name>` during logging.
        :param input_pos_args: input metrics' names, **ordered**
        :param phases:
        """
        super().__init__(name, input_pos_args, phases)

        if input_kwargs is None:
            input_kwargs = {}
        self.input_kwargs = input_kwargs

        self.metrics = nn.ModuleDict({
            self.PHASE_PREFIX+p: metric() for p in phases
        })

    def epoch_update(self, phase, **kwargs):
        if phase not in PHASES:
            raise ValueError(f'phase should be one of `{PHASES}`')

        if phase not in self.phases:
            return

        inputs = [kwargs[k] for k in self.input_keys]
        batch_val = self.metrics[self.PHASE_PREFIX+phase](*inputs, **self.input_kwargs)

    def get_value(self, phase) -> Dict:
        if phase not in PHASES:
            raise ValueError(f'phase should be one of `{PHASES}`')

        if phase not in self.phases:
            return {}

        val = self.metrics[self.PHASE_PREFIX+phase].compute()

        return {f'{self.name}': val}

    def reset(self, phase):
        if phase not in PHASES:
            raise ValueError(f'phase should be one of `{PHASES}`')

        if phase not in self.phases:
            return

        self.metrics[self.PHASE_PREFIX+phase].reset()
