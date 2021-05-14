import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler

class Optimizer(Adam):
    def __init__(self, parameters, lr=0, betas=(0.9, 0.98), eps=1e-9, d_model: int = 512, factor: float = 1.0, warmup_steps: int = 16000):
        super().__init__(parameters, lr=lr, betas=betas, eps=eps)
        self.my_step_num = 0
        self.my_lr = 0.0
        self.my_warmup_steps = warmup_steps
        self.d_model = d_model
        self.my_factor = factor


    @torch.no_grad()
    def step(self, closure=None):
        # calcualte new lr
        self.my_step_num += 1
        self.my_lr = self.my_factor * self.d_model ** (-0.5) * min(self.my_step_num ** (-0.5), self.my_step_num * self.my_warmup_steps ** (-1.5))

        # set new lr
        for param in self.param_groups:
            param['lr'] = self.my_lr

        # forward step call to real optimizer
        super().step(closure=closure)


    def state_dict(self):
        super_dict = super().state_dict()
        super_dict['my_step_num'] = self.my_step_num
        super_dict['my_lr'] = self.my_lr
        super_dict['my_factor'] = self.my_factor
        super_dict['my_warmup_steps'] = self.my_warmup_steps
        return super_dict


    def load_state_dict(self, state_dict):
        self.my_step_num = state_dict['my_step_num']
        self.my_lr = state_dict['my_lr']
        self.my_factor = state_dict['my_factor']
        self.my_warmup_steps = state_dict['my_warmup_steps']
        del state_dict["my_step_num"]
        del state_dict["my_lr"]
        del state_dict["my_factor"]
        del state_dict["my_warmup_steps"]
        super().load_state_dict(state_dict)
