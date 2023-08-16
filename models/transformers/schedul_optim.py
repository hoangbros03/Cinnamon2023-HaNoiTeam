class ScheduleOptimize:
    def __init__(self, optim, init_lr, d_model, n_warmup_steps=4000):
        self._optim = optim
        self._init_lr = init_lr
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def zero_grad(self):
        return self._optim.zero_grad()

    def _get_lr_scale(self):
        return (self.d_model**-0.5) * min(
            self.n_steps**-0.5, self.n_steps * self.n_warmup_steps**-1.5
        )

    def update_and_step(self):
        self._update_lr()
        self._optim.step()

    def _update_lr(self):
        self.n_steps += 1
        lr = self._init_lr * self._get_lr_scale()
        for param_group in self._optim.param_groups:
            param_group["lr"] = lr

    def save_state_dict(self):
        state_dict = {
            "opt": self._optim.state_dict(),
            "lr": self._init_lr,
            "n_warmup_steps": self.n_warmup_steps,
            "d_model": self.d_model,
            "n_steps": self.n_steps,
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.init_lr = state_dict["lr"]
        self._optim.load_state_dict(state_dict["opt"])
        self.d_model = state_dict["d_model"]
        self.n_steps = state_dict["n_steps"]
        self.n_warmup_steps = state_dict["n_warmup_steps"]
