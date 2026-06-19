from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.structural_execution_probe import ProbeError, evaluate_manifest


def evidence(capacity: float, *, kind: str, window: str | None = None):
    record = {
        "status": "PASS",
        "certified": True,
        "capacity_lcb_usd": capacity,
        "sample_count": 120,
        "coverage": 0.99,
    }
    if window is not None:
        record["window"] = window
    if kind == "fill":
        record["mode"] = "shadow"
    return record


def base_manifest():
    return {
        "portfolio": {
            "target_notional_usd": 1_000_000,
            "capacity_safety_multiple": 2.0,
            "selected_window": "00:00-08:00 UTC",
            "required_evidence": ["timed_depth", "fill"],
            "required_gates": ["watch_gate", "risk_gate", "ops_gate"],
            "evidence_requirements": {
                "timed_depth": {
                    "require_certified": True,
                    "require_lcb": True,
                    "require_window": True,
                    "min_samples": 100,
                    "min_coverage": 0.95,
                },
                "fill": {
                    "require_certified": True,
                    "require_lcb": True,
                    "min_samples": 100,
                    "allowed_modes": ["replay", "shadow", "live"],
                },
            },
        },
        "gates": {"watch_gate": "PASS", "risk_gate": "PASS", "ops_gate": "PASS"},
        "sleeves": [
            {
                "name": "cvar_funding_carry",
                "weight": 0.6,
                "research_gate": "PASS",
                "evidence": {
                    "current_depth": {"status": "PASS", "capacity_usd": 1_800_000},
                    "timed_depth": evidence(1_500_000, kind="timed_depth", window="00:00-08:00 UTC"),
                    "fill": evidence(1_400_000, kind="fill"),
                },
            },
            {
                "name": "coin_m_basis",
                "weight": 0.4,
                "research_gate": "PASS",
                "evidence": {
                    "current_depth": {"status": "PASS", "capacity_usd": 1_100_000},
                    "timed_depth": evidence(1_000_000, kind="timed_depth", window="00:00-08:00 UTC"),
                    "fill": evidence(900_000, kind="fill"),
                },
            },
        ],
    }


class ProbeTests(unittest.TestCase):
    def test_opens_when_all_required_evidence_and_capacity_pass(self):
        report = evaluate_manifest(base_manifest())
        self.assertEqual(report["deploy_gate"], "OPEN")
        self.assertEqual(report["deployable_count"], 1)
        self.assertAlmostEqual(report["certified_execution"]["portfolio_capacity_usd"], 2_250_000)
        self.assertEqual(report["certified_execution"]["binding"]["sleeve"], "coin_m_basis")

    def test_missing_fill_fails_closed_even_when_current_depth_is_large(self):
        manifest = base_manifest()
        del manifest["sleeves"][1]["evidence"]["fill"]
        report = evaluate_manifest(manifest)
        self.assertEqual(report["deploy_gate"], "CLOSED")
        self.assertEqual(report["deployable_count"], 0)
        self.assertIsNone(report["certified_execution"]["portfolio_capacity_usd"])
        self.assertIn("MISSING_EVIDENCE", {b["code"] for b in report["blocker_matrix"]})
        self.assertIsNotNone(report["stage_capacity"]["current_depth_proxy"]["portfolio_capacity_usd"])

    def test_low_capacity_identifies_binding_leg(self):
        manifest = base_manifest()
        manifest["sleeves"][1]["evidence"]["fill"]["capacity_lcb_usd"] = 700_000
        report = evaluate_manifest(manifest)
        self.assertEqual(report["deploy_gate"], "CLOSED")
        self.assertAlmostEqual(report["certified_execution"]["portfolio_capacity_usd"], 1_750_000)
        capacity_blocker = next(
            b for b in report["blocker_matrix"]
            if b["code"] == "CAPACITY_BELOW_REQUIRED_MULTIPLE"
        )
        self.assertEqual(capacity_blocker["sleeve"], "coin_m_basis")

    def test_hedge_multiplier_is_applied_before_structural_weight(self):
        manifest = base_manifest()
        sleeve = manifest["sleeves"][1]
        sleeve.pop("evidence")
        sleeve["legs"] = [
            {
                "name": "inverse_future",
                "hedge_multiplier": 1.25,
                "evidence": {
                    "timed_depth": evidence(1_250_000, kind="timed_depth", window="00:00-08:00 UTC"),
                    "fill": evidence(1_125_000, kind="fill"),
                    "current_depth": {"status": "PASS", "capacity_usd": 1_500_000},
                },
            }
        ]
        report = evaluate_manifest(manifest)
        self.assertAlmostEqual(report["certified_execution"]["portfolio_capacity_usd"], 2_250_000)
        self.assertEqual(report["deploy_gate"], "OPEN")

    def test_weight_sum_mismatch_is_a_hard_blocker(self):
        manifest = base_manifest()
        manifest["sleeves"][1]["weight"] = 0.3
        report = evaluate_manifest(manifest)
        self.assertEqual(report["deploy_gate"], "CLOSED")
        self.assertIn("INVALID_WEIGHT_SUM", {b["code"] for b in report["blocker_matrix"]})

    def test_contract_units_are_rejected(self):
        manifest = base_manifest()
        manifest["sleeves"][1]["evidence"]["fill"]["unit"] = "contracts"
        report = evaluate_manifest(manifest)
        self.assertEqual(report["deploy_gate"], "CLOSED")
        self.assertIn("INVALID_CAPACITY_UNIT", {b["code"] for b in report["blocker_matrix"]})

    def test_invalid_target_rejected(self):
        manifest = base_manifest()
        manifest["portfolio"]["target_notional_usd"] = 0
        with self.assertRaises(ProbeError):
            evaluate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
