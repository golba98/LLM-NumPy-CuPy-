import numpy as np
from llm_numpy.backend import array_module, get_backend, to_device
from typing import List, Dict, Union, Tuple, Iterable
from llm_numpy.parameter import Parameter

class AdamW:
    """
    AdamW (Adaptive Moment Estimation with Decoupled Weight Decay).
    theta_t+1 = theta_t * (1 - lr * weight_decay) - lr * m_hat / (sqrt(v_hat) + eps)
    Supports parameter groups (e.g. decay for matrices, no decay for norm scale weights).
    """
    def __init__(
        self,
        params: Union[Iterable[Parameter], List[Dict]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        self.lr = lr
        self.betas = betas
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.param_groups: List[Dict] = []

        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for group in params:
                group_copy = dict(group)
                group_copy.setdefault("lr", self.lr)
                group_copy.setdefault("betas", self.betas)
                group_copy.setdefault("eps", self.eps)
                group_copy.setdefault("weight_decay", self.weight_decay)
                # Deduplicate parameters in group
                p_list = []
                seen = set()
                for p in group_copy["params"]:
                    if p.requires_grad and id(p) not in seen:
                        seen.add(id(p))
                        p_list.append(p)
                group_copy["params"] = p_list
                group_copy["m"] = [array_module(p.data).zeros_like(p.data) for p in p_list]
                group_copy["v"] = [array_module(p.data).zeros_like(p.data) for p in p_list]
                self.param_groups.append(group_copy)
        else:
            p_list = []
            seen = set()
            for p in params:
                if p.requires_grad and id(p) not in seen:
                    seen.add(id(p))
                    p_list.append(p)
            self.param_groups.append({
                "params": p_list,
                "lr": self.lr,
                "betas": self.betas,
                "eps": self.eps,
                "weight_decay": self.weight_decay,
                "m": [array_module(p.data).zeros_like(p.data) for p in p_list],
                "v": [array_module(p.data).zeros_like(p.data) for p in p_list]
            })

    def step(self) -> None:
        """Executes a single optimization step across all parameter groups."""
        self.t += 1
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                g = p.grad
                m = group["m"][i]
                v = group["v"][i]
                # FP16 gradients must be promoted before squaring; otherwise
                # small values underflow to zero and AdamW divides by eps.
                g_for_state = g.astype(m.dtype) if str(g.dtype) != str(m.dtype) else g

                # Update 1st and 2nd moment estimates
                m[...] = b1 * m + (1.0 - b1) * g_for_state
                v[...] = b2 * v + (1.0 - b2) * (g_for_state ** 2)

                # Compute bias-corrected estimates
                m_hat = m / (1.0 - (b1 ** self.t))
                v_hat = v / (1.0 - (b2 ** self.t))

                # Decoupled weight decay
                if wd != 0.0:
                    p.data -= lr * wd * p.data

                # Adaptive gradient update
                p.data -= lr * m_hat / (array_module(p.data).sqrt(v_hat) + eps)

    def zero_grad(self) -> None:
        """Clears accumulated gradients across all parameter groups."""
        for group in self.param_groups:
            for p in group["params"]:
                p.zero_grad()

    def to(self, device: str, dtype=None):
        """Move optimizer moments to the same backend as the parameters."""
        backend = get_backend(device)
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                p.to(device, dtype=dtype)
                group["m"][i] = backend.xp.asarray(group["m"][i])
                group["v"][i] = backend.xp.asarray(group["v"][i])
                if dtype is not None:
                    # Keep AdamW moments in FP32 when parameters use FP16.
                    moment_dtype = np.float32 if np.dtype(dtype) == np.dtype(np.float16) else dtype
                    group["m"][i] = group["m"][i].astype(moment_dtype)
                    group["v"][i] = group["v"][i].astype(moment_dtype)
        return self

    def state_dict(self) -> Dict:
        return {
            "t": self.t,
            "lr": self.lr,
            "betas": self.betas,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "m": [[value.copy() for value in group["m"]] for group in self.param_groups],
            "v": [[value.copy() for value in group["v"]] for group in self.param_groups],
        }

    def load_state_dict(self, state: Dict) -> None:
        self.t = int(state["t"])
        for group, moments_m, moments_v in zip(self.param_groups, state["m"], state["v"]):
            for target, source in zip(group["m"], moments_m):
                target[...] = source
            for target, source in zip(group["v"], moments_v):
                target[...] = source
