from torch.optim.lr_scheduler import _LRScheduler
import torch


class WarmupLinearScheduleNonZero(_LRScheduler):
    """Linear warmup and then linear decay.
    Linearly increases learning rate from 0 to max_lr over `warmup_steps` training steps.
    Linearly decreases learning rate linearly to min_lr over remaining `t_total - warmup_steps` steps.
    """

    def __init__(self, optimizer, warmup_steps, t_total, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.min_lr = min_lr
        super(WarmupLinearScheduleNonZero, self).__init__(
            optimizer, last_epoch=last_epoch
        )

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            lr_factor = float(step) / float(max(1, self.warmup_steps))
        else:
            lr_factor = max(
                0,
                float(self.t_total - step)
                / float(max(1.0, self.t_total - self.warmup_steps)),
            )

        return [
            base_lr * lr_factor if (base_lr * lr_factor) > self.min_lr else self.min_lr
            for base_lr in self.base_lrs
        ]


def batch_iter(dataloader, params):
    for epochId in range(params["num_epochs"]):
        for idx, batch in enumerate(dataloader):
            yield epochId, idx, batch


def load_pretrained_weights_and_adjust_names(model_path):
    d = torch.load(model_path)
    new_params_dict = {}
    for key, value in d.items():
        new_key = key
        if "embeddings" in key:
            new_key = new_key.replace("bert", "vd_net")
        if "gamma" in key:
            new_key = new_key.replace("gamma", "weight")
        if "beta" in key:
            new_key = new_key.replace("beta", "bias")
        if "encoder" in key:
            new_key = new_key.replace("bert.encoder.layer", "vd_net.layers")
        if "pooler" in new_key:
            new_key = new_key.replace("bert", "vd_net")
        if "predictions" in new_key:
            new_key = new_key.replace("cls.predictions", "vd_prediction_head")
        if "decoder" in new_key:
            new_key = new_key.replace("decoder", "lm_classifier")
        if "cls.seq_relationship" in new_key:
            new_key = new_key.replace(
                "cls.seq_relationship", "vd_prediction_head.nsp_classifier"
            )
        new_params_dict[new_key] = value
    return new_params_dict
