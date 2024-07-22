

class CosineAnnealingWarmBootingLR:
    # cawb learning rate scheduler: given the warm booting steps, calculate the learning rate automatically

    def __init__(self, optimizer, epochs=0, eta_min=0.05, steps=[], step_scale=0.8, lf=None, batchs=0, warmup_epoch=0,
                 epoch_scale=1.0, last_epoch=0):
        self.warmup_iters = batchs * warmup_epoch
        self.optimizer = optimizer
        self.eta_min = eta_min
        self.iters = -1
        self.iters_batch = -1
        self.base_lr = [group['lr'] for group in optimizer.param_groups]
        self.step_scale = step_scale
        steps.sort()
        self.steps = [warmup_epoch] + [i for i in steps if (i < epochs and i > warmup_epoch)] + [epochs]
        self.gap = 0
        self.last_epoch = last_epoch
        self.lf = lf
        self.epoch_scale = epoch_scale

        # Initialize epochs and base learning rates
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

    def step(self, external_iter=None):
        self.iters += 1
        if external_iter is not None:
            self.iters = external_iter

        # cos warm boot policy
        iters = self.iters + self.last_epoch
        scale = 1.0
        for i in range(len(self.steps) - 1):
            if (iters <= self.steps[i + 1]):
                self.gap = self.steps[i + 1] - self.steps[i]
                iters = iters - self.steps[i]

                if i != len(self.steps) - 2:
                    self.gap += self.epoch_scale
                break
            scale *= self.step_scale

        if self.lf is None:
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = (scale * lr * (((1 + math.cos(iters * math.pi / self.gap)) / 2) ** 1.0) * (
                        1.0 - self.eta_min) + self.eta_min)
        else:
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = scale * lr * self.lf(iters, self.gap)

        return self.optimizer.param_groups[0]['lr']

    def step_batch(self):
        self.iters_batch += 1

        if self.iters_batch < self.warmup_iters:

            rate = self.iters_batch / self.warmup_iters
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * rate
            return self.optimizer.param_groups[0]['lr']
        else:
            return None


if __name__ == '__main__':
    import torch.nn as nn
    import torch.optim as optim
    from plotly import graph_objects as go
    import math


    def test_CosWarmbootimgLR():
        model = nn.Linear(12, 12)

        lr = 1e-4
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10,
        #                                                            T_mult=2, eta_min=0.001,
        #                                                            last_epoch=-1)
        scheduler = CosineAnnealingWarmBootingLR(optimizer, epochs=1000, eta_min=1e-5, steps=[500, ], step_scale=0.01,
                                                 last_epoch=0)

        lr_list = []

        lr_list.append(optimizer.param_groups[0]['lr'])
        last_lr = optimizer.param_groups[0]['lr']

        for e in range(2000):
            scheduler.step()
            # print(scheduler.__dict__)
            # input()
            # if optimizer.param_groups[0]['lr'] > last_lr:
            #     for p in optimizer.param_groups:
            #         p["lr"] = lr * 0.5
            #         p["initial_lr"] = lr * 0.5
            #     scheduler.base_lrs = lr * 0.5
            #     lr *= 0.5
            lr_list.append(optimizer.param_groups[0]['lr'])
            last_lr = optimizer.param_groups[0]['lr']

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(lr_list))), y=lr_list))
        fig.show()


    test_CosWarmbootimgLR()
