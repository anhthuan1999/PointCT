from torch.optim.lr_scheduler import LambdaLR, StepLR, OneCycleLR
import torch.optim as optim

class LambdaStepLR(LambdaLR):

  def __init__(self, optimizer, lr_lambda, last_step=-1):
    super(LambdaStepLR, self).__init__(optimizer, lr_lambda, last_step)

  @property
  def last_step(self):
    """Use last_epoch for the step counter"""
    return self.last_epoch

  @last_step.setter
  def last_step(self, v):
    self.last_epoch = v

class MultiStepWithWarmup(LambdaStepLR):
  def __init__(self, optimizer, milestones, gamma=0.1, warmup='linear', warmup_iters=1500, warmup_ratio=1e-6, last_step=-1):

    assert warmup == 'linear'
    def multi_step_with_warmup(s):
      factor = 1.0
      for i in range(len(milestones)):
        if s < milestones[i]:
          break
        factor *= gamma
      #print(s)
      if s <= warmup_iters:
        warmup_coeff = 1 - (1 - s / warmup_iters) * (1 - warmup_ratio)
      else:
        warmup_coeff = 1.0
      return warmup_coeff * factor

    super(MultiStepWithWarmup, self).__init__(optimizer, multi_step_with_warmup, last_step)