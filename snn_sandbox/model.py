from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any

import torch


@dataclass
class ArithmeticSNNConfig:
    input_neurons: int = 4
    hidden_neurons: int = 96
    output_neurons: int = 3
    episode_steps: int = 28
    input_rate: float = 0.72
    background_rate: float = 0.0
    hidden_threshold: float = 0.34
    hidden_winners: int = 12
    output_threshold: float = 0.18
    trace_decay: float = 0.88
    ih_ltp: float = 0.003
    ih_ltd: float = 0.001
    ih_column_sum: float = 1.0
    ho_lr: float = 0.032
    ho_competitor_depress: float = 0.018
    weight_decay: float = 0.0002
    min_weight: float = 0.0
    max_ih_weight: float = 1.0
    max_ho_weight: float = 1.4
    seed: int = 7


PATTERNS: tuple[tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))


class ArithmeticSTDPSNN:
    """One-bit addition SNN using local STDP and reward-modulated STDP.

    The model intentionally avoids autograd and backpropagation. All plasticity is
    applied directly to synapses from spike traces and a scalar correctness signal.
    """

    def __init__(
        self,
        config: ArithmeticSNNConfig | None = None,
        device: str = "cuda",
        allow_cpu: bool = False,
    ) -> None:
        self.config = config or ArithmeticSNNConfig()
        if device == "cuda" and not torch.cuda.is_available():
            if not allow_cpu:
                raise RuntimeError(
                    "CUDA is not available. Run on the desktop GPU host, install a CUDA-enabled "
                    "PyTorch build, or pass --allow-cpu for a slow smoke test."
                )
            device = "cpu"

        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.config.seed)
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.config.seed = seed
        self.generator.manual_seed(self.config.seed)
        c = self.config
        self.input_hidden = self._initial_input_hidden()
        self.hidden_output = torch.rand(
            (c.hidden_neurons, c.output_neurons),
            device=self.device,
            generator=self.generator,
        ) * 0.025
        self.pre_trace = torch.zeros(c.input_neurons, device=self.device)
        self.hidden_trace = torch.zeros(c.hidden_neurons, device=self.device)
        self.recent_spikes = torch.zeros((96, c.hidden_neurons), device=self.device)
        self.spike_cursor = 0
        self.trials = 0
        self.correct = 0
        self.pattern_counts = torch.zeros(4, device=self.device)
        self.pattern_correct = torch.zeros(4, device=self.device)
        self.confusion = torch.zeros((c.output_neurons, c.output_neurons), device=self.device)
        self.last_result: dict[str, Any] = {}
        self.started_at = time.time()

    def train_episodes(self, count: int) -> None:
        for _ in range(max(0, count)):
            pattern_index = int(torch.randint(0, len(PATTERNS), (1,), generator=self.generator, device=self.device).item())
            self.run_episode(PATTERNS[pattern_index], pattern_index, learn=True)

    def evaluate_all(self) -> list[dict[str, Any]]:
        results = []
        for pattern_index, pattern in enumerate(PATTERNS):
            results.append(self.run_episode(pattern, pattern_index, learn=False))
        return results

    def run_episode(
        self,
        pattern: tuple[int, int],
        pattern_index: int,
        learn: bool,
    ) -> dict[str, Any]:
        c = self.config
        target = pattern[0] + pattern[1]
        input_template = self._encode_pattern(pattern)
        hidden_count = torch.zeros(c.hidden_neurons, device=self.device)
        output_drive = torch.zeros(c.output_neurons, device=self.device)

        for _ in range(c.episode_steps):
            if learn:
                rand = torch.rand(c.input_neurons, device=self.device, generator=self.generator)
                input_rates = torch.where(
                    input_template > 0,
                    torch.full_like(input_template, c.input_rate),
                    torch.full_like(input_template, c.background_rate),
                )
                input_spikes = (rand < input_rates).float()
            else:
                input_spikes = input_template

            hidden_current = input_spikes @ self.input_hidden
            adaptive_noise = torch.rand(c.hidden_neurons, device=self.device, generator=self.generator) * 0.035 if learn else 0.0
            hidden_spikes = self._sparse_hidden_spikes(hidden_current + adaptive_noise)

            output_current = hidden_spikes @ self.hidden_output
            output_spikes = (output_current > c.output_threshold).float()
            output_drive += output_current + (0.05 * output_spikes)
            hidden_count += hidden_spikes

            if learn:
                self._apply_input_hidden_stdp(input_spikes, hidden_spikes)

            if learn:
                self.recent_spikes[self.spike_cursor % self.recent_spikes.shape[0]] = hidden_spikes
                self.spike_cursor += 1

        prediction = int(torch.argmax(output_drive).item())
        correct = prediction == target

        if learn:
            self._apply_hidden_output_rstdp(hidden_count / c.episode_steps, target, prediction, correct)
            self.trials += 1
            self.correct += int(correct)
            self.pattern_counts[pattern_index] += 1
            self.pattern_correct[pattern_index] += int(correct)
            self.confusion[target, prediction] += 1

        margin = self._margin(output_drive, target)
        result = {
            "pattern": f"{pattern[0]}+{pattern[1]}",
            "target": target,
            "prediction": prediction,
            "correct": correct,
            "outputDrive": output_drive.detach().cpu().round(decimals=3).tolist(),
            "activeHidden": int((hidden_count > 0).sum().item()),
            "meanHiddenRate": float((hidden_count / c.episode_steps).mean().item()),
            "margin": margin,
        }
        if learn:
            self.last_result = result
        return result

    def snapshot(self, running: bool, episodes_per_tick: int, seq: int) -> dict[str, Any]:
        eval_results = self.evaluate_all()
        eval_correct = sum(1 for item in eval_results if item["correct"])
        accuracy = self.correct / self.trials if self.trials else 0.0
        per_pattern = torch.where(
            self.pattern_counts > 0,
            self.pattern_correct / torch.clamp(self.pattern_counts, min=1),
            torch.zeros_like(self.pattern_counts),
        )
        recent = torch.roll(self.recent_spikes, shifts=-int(self.spike_cursor % self.recent_spikes.shape[0]), dims=0)
        visible_cols = min(48, self.config.hidden_neurons)
        weights = self.hidden_output.detach().cpu()
        ih = self.input_hidden.detach().cpu()
        return {
            "seq": seq,
            "running": running,
            "device": str(self.device),
            "cudaAvailable": torch.cuda.is_available(),
            "gpuName": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "episodesPerTick": episodes_per_tick,
            "trials": self.trials,
            "accuracy": round(accuracy, 4),
            "evalAccuracy": round(eval_correct / len(eval_results), 4),
            "eval": eval_results,
            "lastResult": self.last_result,
            "perPatternAccuracy": per_pattern.detach().cpu().round(decimals=4).tolist(),
            "confusion": self.confusion.detach().cpu().int().tolist(),
            "spikeRaster": recent[:, :visible_cols].detach().cpu().int().tolist(),
            "hiddenActivity": recent.mean(dim=0)[:visible_cols].detach().cpu().round(decimals=4).tolist(),
            "hiddenOutputWeights": weights.round(decimals=4).tolist(),
            "inputHiddenSample": ih[:, :visible_cols].round(decimals=4).tolist(),
            "meanInputHiddenWeight": round(float(self.input_hidden.mean().item()), 4),
            "meanHiddenOutputWeight": round(float(self.hidden_output.mean().item()), 4),
            "architecture": {
                "inputs": self.config.input_neurons,
                "hidden": self.config.hidden_neurons,
                "outputs": self.config.output_neurons,
                "episodeSteps": self.config.episode_steps,
                "learning": "input->hidden local STDP; hidden->output reward-modulated STDP; no autograd/backprop",
            },
            "config": asdict(self.config),
            "uptimeSeconds": round(time.time() - self.started_at, 1),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "input_hidden": self.input_hidden.detach().cpu(),
            "hidden_output": self.hidden_output.detach().cpu(),
            "trials": self.trials,
            "correct": self.correct,
            "pattern_counts": self.pattern_counts.detach().cpu(),
            "pattern_correct": self.pattern_correct.detach().cpu(),
            "confusion": self.confusion.detach().cpu(),
            "last_result": self.last_result,
        }

    def load_state_dict(self, data: dict[str, Any]) -> None:
        self.config = ArithmeticSNNConfig(**data["config"])
        self.input_hidden = data["input_hidden"].to(self.device)
        self.hidden_output = data["hidden_output"].to(self.device)
        self.trials = int(data.get("trials", 0))
        self.correct = int(data.get("correct", 0))
        self.pattern_counts = data.get("pattern_counts", torch.zeros(4)).to(self.device)
        self.pattern_correct = data.get("pattern_correct", torch.zeros(4)).to(self.device)
        self.confusion = data.get("confusion", torch.zeros((3, 3))).to(self.device)
        self.last_result = data.get("last_result", {})
        c = self.config
        self.pre_trace = torch.zeros(c.input_neurons, device=self.device)
        self.hidden_trace = torch.zeros(c.hidden_neurons, device=self.device)
        self.recent_spikes = torch.zeros((96, c.hidden_neurons), device=self.device)
        self.spike_cursor = 0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: Path) -> None:
        data = torch.load(path, map_location="cpu")
        self.load_state_dict(data)

    def _encode_pattern(self, pattern: tuple[int, int]) -> torch.Tensor:
        encoded = torch.zeros(self.config.input_neurons, device=self.device)
        encoded[0 if pattern[0] == 0 else 1] = 1
        encoded[2 if pattern[1] == 0 else 3] = 1
        return encoded

    def _initial_input_hidden(self) -> torch.Tensor:
        c = self.config
        weights = torch.rand(
            (c.input_neurons, c.hidden_neurons),
            device=self.device,
            generator=self.generator,
        ) * 0.05
        for hidden_index in range(c.hidden_neurons):
            pattern = PATTERNS[hidden_index % len(PATTERNS)]
            encoded = self._encode_pattern(pattern)
            preferred = torch.rand(c.input_neurons, device=self.device, generator=self.generator) * 0.16 + 0.42
            non_preferred = torch.rand(c.input_neurons, device=self.device, generator=self.generator) * 0.05
            weights[:, hidden_index] = torch.where(encoded > 0, preferred, non_preferred)
        return weights

    def _sparse_hidden_spikes(self, hidden_current: torch.Tensor) -> torch.Tensor:
        c = self.config
        eligible = hidden_current > c.hidden_threshold
        spikes = torch.zeros_like(hidden_current)
        if not bool(eligible.any().item()):
            return spikes
        winner_count = min(c.hidden_winners, int(eligible.sum().item()))
        winner_indices = torch.topk(hidden_current.masked_fill(~eligible, -1e9), winner_count).indices
        spikes[winner_indices] = 1.0
        return spikes

    def _apply_input_hidden_stdp(self, input_spikes: torch.Tensor, hidden_spikes: torch.Tensor) -> None:
        c = self.config
        self.pre_trace.mul_(c.trace_decay).add_(input_spikes)
        self.hidden_trace.mul_(c.trace_decay).add_(hidden_spikes)
        ltp = torch.outer(self.pre_trace, hidden_spikes) * c.ih_ltp
        ltd = torch.outer(input_spikes, self.hidden_trace) * c.ih_ltd
        self.input_hidden.add_(ltp - ltd)
        self.input_hidden.mul_(1.0 - c.weight_decay)
        self.input_hidden.clamp_(c.min_weight, c.max_ih_weight)
        column_sum = self.input_hidden.sum(dim=0, keepdim=True).clamp_min(1e-6)
        self.input_hidden.mul_(c.ih_column_sum / column_sum)

    def _apply_hidden_output_rstdp(
        self,
        hidden_rates: torch.Tensor,
        target: int,
        prediction: int,
        correct: bool,
    ) -> None:
        c = self.config
        target_vec = torch.zeros(c.output_neurons, device=self.device)
        target_vec[target] = 1.0
        pred_vec = torch.zeros(c.output_neurons, device=self.device)
        pred_vec[prediction] = 1.0
        if correct:
            modulation = 0.18 * target_vec
        else:
            modulation = target_vec - pred_vec
        delta = torch.outer(hidden_rates, modulation) * c.ho_lr
        self.hidden_output.add_(delta)
        self.hidden_output.mul_(1.0 - c.weight_decay)
        self.hidden_output.clamp_(c.min_weight, c.max_ho_weight)

    def _margin(self, output_drive: torch.Tensor, target: int) -> float:
        target_value = output_drive[target]
        masked = output_drive.clone()
        masked[target] = -1e9
        return round(float((target_value - masked.max()).item()), 4)
