import torch

class SharedEve(torch.optim.Optimizer):

    """
    Following to Hiroaki Hayashi, Jayanth Koushik, Graham Neubig
    "Eve: A Gradient Based Optimization Method with Locally and Globally Adaptive Learning Rates" (arXiv:1611.01505v1)
    """

    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, b3=0.999, eps=1e-8, c=10.0, f_min=0.0, weight_decay=0.0):

        defaults = dict()
        defaults['eps'] = torch.tensor(eps, dtype=torch.float32, requires_grad=False)
        defaults['b1'] = torch.tensor(b1, dtype=torch.float32, requires_grad=False)
        defaults['b2'] = torch.tensor(b2, dtype=torch.float32, requires_grad=False)
        defaults['b3'] = torch.tensor(b3, dtype=torch.float32, requires_grad=False)
        defaults['lr'] = torch.tensor(lr, dtype=torch.float32, requires_grad=False)
        defaults['c'] = torch.tensor(c, dtype=torch.float32, requires_grad=False)
        defaults['c_inv'] = torch.tensor(1 / c, dtype=torch.float32, requires_grad=False)
        defaults['f_min'] = torch.tensor(f_min, dtype=torch.float32, requires_grad=False)
        defaults['step'] = torch.tensor(0, dtype=torch.int32, requires_grad=False)
        defaults['b1_pow'] = torch.tensor(1, dtype=torch.float32, requires_grad=False)
        defaults['b2_pow'] = torch.tensor(1, dtype=torch.float32, requires_grad=False)
        defaults['d'] = torch.tensor(1, dtype=torch.float32, requires_grad=False)
        defaults['f_old'] = torch.tensor(0, dtype=torch.float32, requires_grad=False)
        defaults['weight_decay'] = torch.tensor(weight_decay, dtype=torch.float32, requires_grad=False)

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

            f_ = loss.data
            if group['step'].is_nonzero():
                d = abs(f_ - group['f_old']) / (min(f_, group['f_old']) - group['f_min'])
                d = max(min(d, group['c']), group['c_inv'])
                group['d'].fill_(group['b3'] * group['d'] + (1.0 - group['b3']) * d)
            else:
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