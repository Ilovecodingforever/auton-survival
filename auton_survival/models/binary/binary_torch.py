import torch.nn as nn
from auton_survival.models.cph.dcph_torch import DeepCoxPHTorch


class BinarySurvivalClassifierTorch(DeepCoxPHTorch):
  def __init__(self, inputdim, layers=None, optimizer='Adam',
               survival_estimator='km', n_bins=20):
    super(BinarySurvivalClassifierTorch, self).__init__(inputdim,
                                                        layers=layers,
                                                        optimizer=optimizer)
    self.survival_estimator = survival_estimator
    self.n_bins = n_bins

  def _init_coxph_layers(self, lastdim):
    self.expert = nn.Linear(lastdim, 1, bias=True)
