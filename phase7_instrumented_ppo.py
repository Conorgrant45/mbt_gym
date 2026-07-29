"""
phase7_instrumented_ppo.py
--------------------------------
Phase7InstrumentedPPO: subclasses phase5_instrumented_ppo.InstrumentedPPO
(itself a byte-for-byte copy of SB3 2.8.0's PPO.train() with read-only
per-update diagnostics added -- see that module's docstring) and appends,
to the SAME per-update record, the few additional fields Phase 7's brief
asks for that Phase 5's version did not separately expose: log_std and
action_std PER DIMENSION (bid/ask) rather than only their mean, the
current learning rate, and the raw total loss value SB3 itself logs.

No PPO update math is touched here -- this class only reads already-
computed values (self.policy.log_std, self.policy.optimizer's learning
rate, self.logger's own just-recorded values) after calling
super().train(), which performs the actual (unmodified) optimisation step.
"""
import numpy as np

from phase5_instrumented_ppo import InstrumentedPPO


class Phase7InstrumentedPPO(InstrumentedPPO):
    def train(self) -> None:
        super().train()

        log_std = self.policy.log_std.detach().numpy().copy()
        action_std = np.exp(log_std)
        current_lr = float(self.policy.optimizer.param_groups[0]["lr"])
        total_loss = self.logger.name_to_value.get("train/loss", float("nan"))

        rec = self.update_records[-1]
        rec["log_std_bid"] = float(log_std[0])
        rec["log_std_ask"] = float(log_std[1])
        rec["action_std_bid"] = float(action_std[0])
        rec["action_std_ask"] = float(action_std[1])
        rec["learning_rate"] = current_lr
        rec["total_loss"] = float(total_loss)
