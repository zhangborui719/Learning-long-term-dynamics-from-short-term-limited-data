"""The flow-map network and its training loop.

Architecture follows the assignment's specification: a fully connected map from
the 40-dimensional state to the 40-dimensional next state, with **two hidden
layers of 80 units each**,

    40 -> [Linear + tanh] x 2 (80 units) -> Linear -> 40,

for 13 000 trainable parameters in total.  ``tanh`` is chosen over ``ReLU``
because a flow map is a smooth diffeomorphism: the target is ``C^1`` in the
input, and a piecewise-linear activation would introduce kinks the true map does
not have.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import torch
from torch import nn

LOGGER = logging.getLogger("flowmap.model")

ActivationName = Literal["tanh", "silu", "gelu", "relu"]
ScheduleName = Literal["cosine", "linear", "constant"]


# ---------------------------------------------------------------------------
# learning-rate annealing
# ---------------------------------------------------------------------------
def lr_multiplier(
    epoch: int,
    epochs: int,
    warmup_epochs: int = 0,
    final_fraction: float = 1e-3,
    shape: ScheduleName = "cosine",
) -> float:
    """Fraction of the peak learning rate to apply at ``epoch``.

    The schedule is ``warmup (optional) -> decay -> final_fraction``.  Decay to a
    small but non-zero value (``final_fraction = 1e-3``) rather than to exactly
    zero keeps the last epochs from freezing in a high-curvature region, which
    matters here because the loss surface of a chaotic flow map has very large
    curvature in the advection-dominated directions.

    Args:
        epoch: zero-based epoch index.
        epochs: total number of epochs.
        warmup_epochs: length of the linear warmup from ``~0`` to the peak.
        final_fraction: multiplier reached at the last epoch.
        shape: ``"cosine"``, ``"linear"`` or ``"constant"`` (ablation control).

    Returns:
        A multiplier in ``[0, 1]`` that is never below ``final_fraction``.
    """
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if not 0.0 <= final_fraction < 1.0:
        raise ValueError(f"final_fraction must lie in [0, 1), got {final_fraction}")
    warmup_epochs = int(max(0, min(warmup_epochs, epochs - 1)))

    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs

    span = max(1, epochs - warmup_epochs)
    progress = min(1.0, (epoch - warmup_epochs) / span)
    if shape == "cosine":
        base = 0.5 * (1.0 + math.cos(math.pi * progress))
    elif shape == "linear":
        base = 1.0 - progress
    elif shape == "constant":
        base = 1.0
    else:
        raise ValueError(f"unknown schedule shape {shape!r}")
    return final_fraction + (1.0 - final_fraction) * base


def build_scheduler(
    optimiser: torch.optim.Optimizer,
    epochs: int,
    peak_lr: float,
    schedule: ScheduleName = "cosine",
    final_fraction: float = 1e-3,
    warmup_fraction: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Attach an annealing schedule to ``optimiser``.

    Args:
        optimiser: the optimiser to anneal.
        epochs: total number of epochs (the schedule horizon).
        peak_lr: starting learning rate; only used for validation here, the
            multiplier is applied to whatever the optimiser was constructed with.
        schedule: decay shape, see :func:`lr_multiplier`.
        final_fraction: terminal multiplier.
        warmup_fraction: warmup length as a fraction of ``epochs``.

    Returns:
        The scheduler; call ``.step()`` once per epoch.
    """
    if peak_lr <= 0:
        raise ValueError(f"peak_lr must be positive, got {peak_lr}")
    warmup_epochs = int(round(warmup_fraction * epochs))
    return torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lambda epoch: lr_multiplier(epoch, epochs, warmup_epochs, final_fraction, schedule),
    )


def _activation(name: ActivationName) -> nn.Module:
    """Return a fresh activation module by name."""
    table: dict[str, nn.Module] = {
        "tanh": nn.Tanh(),
        "silu": nn.SiLU(),
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
    }
    if name not in table:
        raise ValueError(f"unknown activation {name!r}; expected one of {sorted(table)}")
    return table[name]


class FlowMapMLP(nn.Module):
    """Multilayer perceptron approximating the one-step flow map.

    Args:
        n_dim: state dimension (40 for the F=64 Lorenz-96 system).
        hidden_width: width of every hidden layer.
        n_hidden_layers: number of hidden layers.
        activation: name of the pointwise nonlinearity.
        head_gain: orthogonal initialisation gain of the output layer.  A small
            value starts the map near zero output, i.e. near the training mean,
            which avoids an initial explosion when the state amplitude is large.
    """

    def __init__(
        self,
        n_dim: int,
        hidden_width: int = 80,
        n_hidden_layers: int = 2,
        activation: ActivationName = "tanh",
        head_gain: float = 0.1,
    ) -> None:
        super().__init__()
        if n_dim <= 0:
            raise ValueError(f"n_dim must be positive, got {n_dim}")
        if hidden_width <= 0:
            raise ValueError(f"hidden_width must be positive, got {hidden_width}")
        if n_hidden_layers < 1:
            raise ValueError(f"n_hidden_layers must be >= 1, got {n_hidden_layers}")

        self.n_dim = int(n_dim)
        self.hidden_width = int(hidden_width)
        self.n_hidden_layers = int(n_hidden_layers)
        self.activation_name = activation

        blocks: list[nn.Module] = []
        in_features = self.n_dim
        for _ in range(self.n_hidden_layers):
            blocks.append(nn.Linear(in_features, self.hidden_width))
            blocks.append(_activation(activation))
            in_features = self.hidden_width
        self.body = nn.Sequential(*blocks)

        self.head = nn.Linear(in_features, self.n_dim)
        nn.init.orthogonal_(self.head.weight, gain=head_gain)
        nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map a batch of standardised states to standardised targets."""
        return self.head(self.body(features))

    @property
    def n_parameters(self) -> int:
        """Number of trainable scalar parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        """One-line architecture summary for reports."""
        return (
            f"MLP {self.n_dim} -> "
            + f"[{self.hidden_width} {self.activation_name}] x {self.n_hidden_layers}"
            + f" -> {self.n_dim}, {self.n_parameters} trainable parameters"
        )


@dataclass
class TrainingHistory:
    """Per-epoch optimisation record.

    Attributes:
        epoch: zero-based epoch index.
        train_loss: mean squared error on the training pairs (network space).
        test_loss: mean squared error on the held-out tail, recorded for
            diagnostics only - it is never used to select the model.
        learning_rate: learning rate applied during the epoch.
        grad_norm: global gradient L2 norm *before* the optimiser step.
        extra: additional per-epoch series produced by an evaluation hook (e.g.
            the one-step NRMSE in *state* space, which is the only metric that is
            comparable between target codecs).
    """

    epoch: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    test_loss: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    extra: dict[str, list[float]] = field(default_factory=dict)

    def as_arrays(self) -> dict[str, np.ndarray]:
        """Return the record as plain numpy arrays, base series first."""
        table = {
            "epoch": np.asarray(self.epoch),
            "train_loss": np.asarray(self.train_loss),
            "test_loss": np.asarray(self.test_loss),
            "learning_rate": np.asarray(self.learning_rate),
            "grad_norm": np.asarray(self.grad_norm),
        }
        for name, values in self.extra.items():
            table[name] = np.asarray(values)
        return table


def train_flow_map(
    model: FlowMapMLP,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    epochs: int = 2000,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    seed: int = 0,
    log_every: int = 200,
    schedule: ScheduleName = "cosine",
    final_fraction: float = 1e-3,
    warmup_fraction: float = 0.0,
    eval_hook: Callable[[FlowMapMLP], dict[str, float]] | None = None,
) -> TrainingHistory:
    """Fit ``model`` by mini-batch AdamW with an annealed learning rate.

    The schedule is ``optional warmup -> decay -> final_fraction * lr`` (see
    :func:`build_scheduler`).  Annealing to a small residual value rather than
    stopping at a fixed epoch removes the need to tune a stopping rule: the data
    are deterministic, so there is no label noise to overfit, and the only risk
    is fitting the attractor's *visit distribution* rather than the map.

    Args:
        model: the network to train (modified in place).
        train_x: standardised inputs, shape ``(n, d)``.
        train_y: encoded targets, shape ``(n, d)``.
        test_x: standardised held-out inputs, shape ``(m, d)``.
        test_y: encoded held-out targets, shape ``(m, d)``.
        epochs: number of passes over the training split.
        batch_size: mini-batch size.
        learning_rate: peak Adam learning rate (the start of the anneal).
        weight_decay: decoupled L2 penalty (AdamW).
        seed: seed for mini-batch shuffling.
        log_every: log interval in epochs.
        schedule: ``"cosine"``, ``"linear"`` or ``"constant"``.
        final_fraction: terminal learning-rate multiplier.
        warmup_fraction: warmup length as a fraction of ``epochs``.
        eval_hook: optional callable returning named scalars for the current
            model; recorded every epoch in :attr:`TrainingHistory.extra`.  Used
            to log state-space accuracy on a grid that is identical across
            target codecs.

    Returns:
        The populated :class:`TrainingHistory`.
    """
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if train_x.shape[0] == 0:
        raise ValueError("training split is empty")

    generator = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device

    x_train = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(train_y, dtype=torch.float32, device=device)
    x_test = torch.as_tensor(test_x, dtype=torch.float32, device=device)
    y_test = torch.as_tensor(test_y, dtype=torch.float32, device=device)

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    schedule_fn = build_scheduler(
        optimiser,
        epochs=epochs,
        peak_lr=learning_rate,
        schedule=schedule,
        final_fraction=final_fraction,
        warmup_fraction=warmup_fraction,
    )
    loss_fn = nn.MSELoss()

    history = TrainingHistory()
    n = x_train.shape[0]

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n, generator=generator)
        epoch_loss = 0.0
        epoch_grad = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            batch = permutation[start : start + batch_size]
            prediction = model(x_train[batch])
            loss = loss_fn(prediction, y_train[batch])

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            total_norm = torch.sqrt(
                sum(
                    (p.grad.detach() ** 2).sum()
                    for p in model.parameters()
                    if p.grad is not None
                )
            )
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimiser.step()

            epoch_loss += float(loss.detach())
            epoch_grad += float(total_norm)
            n_batches += 1

        schedule_fn.step()
        model.eval()
        with torch.no_grad():
            test_loss = float(loss_fn(model(x_test), y_test))
            train_eval = float(loss_fn(model(x_train), y_train))

        history.epoch.append(epoch)
        history.train_loss.append(train_eval)
        history.test_loss.append(test_loss)
        history.learning_rate.append(float(schedule_fn.get_last_lr()[0]))
        history.grad_norm.append(epoch_grad / max(1, n_batches))
        if eval_hook is not None:
            for name, value in eval_hook(model).items():
                history.extra.setdefault(name, []).append(float(value))

        if log_every > 0 and (epoch % log_every == 0 or epoch == epochs - 1):
            LOGGER.info(
                "epoch %5d  train %.6e  test %.6e  lr %.2e  |grad| %.3e",
                epoch,
                train_eval,
                test_loss,
                history.learning_rate[-1],
                history.grad_norm[-1],
            )

    return history


@torch.no_grad()
def predict(model: FlowMapMLP, features: np.ndarray) -> np.ndarray:
    """Run the network on a numpy batch and return numpy output."""
    model.eval()
    device = next(model.parameters()).device
    tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    return model(tensor).cpu().numpy().astype(np.float64)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Per-layer trainable parameter counts, for the architecture table."""
    table: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            table[name] = int(parameter.numel())
    return table
