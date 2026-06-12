from __future__ import annotations

from .model import ArithmeticSTDPSNN


def main() -> None:
    model = ArithmeticSTDPSNN(device="cuda", allow_cpu=True)
    model.train_episodes(200)
    snapshot = model.snapshot(running=False, episodes_per_tick=1, seq=1)
    print(
        {
            "device": snapshot["device"],
            "trials": snapshot["trials"],
            "accuracy": snapshot["accuracy"],
            "evalAccuracy": snapshot["evalAccuracy"],
            "lastResult": snapshot["lastResult"],
        }
    )


if __name__ == "__main__":
    main()
