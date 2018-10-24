import torch

class SharedEve(torch.optim.Optimizer):

    """
    Based on Jayanth Koushik, Hiroaki Hayashi "Improving Stochastic Gradient Descent with Feedback" (arXiv:1611.01505v1)
    """

    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, b3=0.999, eps=1e-8, k_inf=0.1, k_sup=10.0, weight_decay=0.0):

        defaults = dict()
        defaults['eps'] = torch.tensor(eps, dtype=torch.float32, requires_grad=False)
        defaults['b1'] = torch.tensor(b1, dtype=torch.float32, requires_grad=False)
        defaults['b2'] = torch.tensor(b2, dtype=torch.float32, requires_grad=False)
        defaults['b3'] = torch.tensor(b3, dtype=torch.float32, requires_grad=False)
        defaults['lr'] = torch.tensor(lr, dtype=torch.float32, requires_grad=False)
        defaults['k_inf_add'] = torch.tensor(k_inf + 1.0, dtype=torch.float32, requires_grad=False)
        defaults['k_sup_add'] = torch.tensor(k_sup + 1.0, dtype=torch.float32, requires_grad=False)
        defaults['k_inf_inv'] = 1 / defaults['k_inf_add']
        defaults['k_sup_inv'] = 1 / defaults['k_sup_add']

        defaults['step'] = torch.tensor(0, dtype=torch.int32, requires_grad=False)
        defaults['b1_pow'] = torch.tensor(1, dtype=torch.float32, requires_grad=False)
        defaults['b2_pow'] = torch.tensor(1, dtype=torch.float32, requires_grad=False)
        defaults['d'] = torch.tensor(1, dtype=torch.float32, requires_grad=False)
        defaults['f_old'] = torch.tensor(0, dtype=torch.float32, requires_grad=False)
        defaults['weight_decay'] = torch.tensor(weight_decay, dtype=torch.float32, requires_grad=False)
        defaults['clip'] = torch.tensor(clip, dtype=torch.float32, requires_grad=False)

        [defaults[k].share_memory_() for k in ['step', 'b1_pow', 'b2_pow', 'd', 'f_old']]
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['exp_avg'] = torch.zeros(p.shape, dtype=torch.float32, requires_grad=False)
                state['exp_avg_sq'] = torch.zeros(p.shape, dtype=torch.float32, requires_grad=False)
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            group['b1_pow'].mul_(group['b1'])
            group['b2_pow'].mul_(group['b2'])

            if group['step'].is_nonzero():
                if loss.data.ge(group['f_old']).is_nonzero():
                    d_1 = group['k_inf_add']
                    d_2 = group['k_sup_add']
                else:
                    d_1 = group['k_sup_inv']
                    d_2 = group['k_inf_inv']

                c = min(max(d_1, (loss.data / group['f_old'])), d_2)
                f_ = c * group['f_old']
                r = abs(c - 1.0) / min(c, 1.0)
                group['d'].fill_(group['b3'] * group['d'] + (1.0 - group['b3']) * r)
            else:
                f_ = loss.data
                group['d'].fill_(1.0)

            group['f_old'].fill_(f_)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if group['weight_decay'] != 0.0:
                    grad = grad.add(group['weight_decay'], p.data)

                state = self.state[p]
                state['exp_avg'][:] = group['b1'] * state['exp_avg'] + (1.0 - group['b1']) * grad
                state['exp_avg_sq'][:] = group['b2'] * state['exp_avg_sq'] + (1.0 - group['b2']) * (grad ** 2)

                b2_decay = (1.0 - group['b2_pow']).sqrt()
                alpha = group['lr'] / group['d'] * (b2_decay / (1.0 - group['b1_pow']))

                p.data.addcdiv_(-alpha, state['exp_avg'], state['exp_avg_sq'].sqrt() + group['eps'])
            
            group['step'].add_(1)

        return loss

class SharedAdam(torch.optim.Adam):
    """
    Copied from MorvanZhou/pytorch-A3C For a benchmark 
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.9), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
