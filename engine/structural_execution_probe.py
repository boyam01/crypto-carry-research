"""Fail-closed structural execution-capacity probe.

The probe is deliberately narrow:
- reads existing JSON artifacts only;
- never calls an exchange/API;
- never runs or re-runs a backtest;
- converts sleeve/leg execution capacity into portfolio-equivalent capacity;
- refuses to turn missing timed-depth/fill evidence into a deployable PASS.

Capacity identity
-----------------
For portfolio notional N, sleeve s with gross structural weight w_s, and leg l
with hedge multiplier h_sl, the leg must execute N * |w_s| * |h_sl|. Therefore
an observed leg capacity C_sl implies portfolio capacity

    C_portfolio = min(C_sl / (|w_s| * |h_sl|)).

A deployable capacity uses the lower-confidence-bound capacity from every
required execution evidence family and takes the minimum across evidence,
legs, and sleeves.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PASS = "PASS"
FAIL = "FAIL"
MISSING = "MISSING"
UNKNOWN = "UNKNOWN"

_PASS_VALUES = {"PASS", "OPEN", "OK", "TRUE", "1"}
_FAIL_VALUES = {"FAIL", "FAILED", "CLOSED", "BLOCKED", "FALSE", "0"}
_MISSING_VALUES = {"MISSING", "ABSENT", "NONE", "NULL", ""}


class ProbeError(ValueError):
    """Raised for malformed manifests or artifacts."""


def _finite_nonnegative(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ProbeError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(out) or out < 0:
        raise ProbeError(f"{field} must be finite and non-negative, got {value!r}")
    return out


def _status(value: Any) -> str:
    if isinstance(value, bool):
        return PASS if value else FAIL
    if value is None:
        return MISSING
    text = str(value).strip().upper()
    if text in _PASS_VALUES:
        return PASS
    if text in _FAIL_VALUES:
        return FAIL
    if text in _MISSING_VALUES:
        return MISSING
    return UNKNOWN


def _json_pointer(document: Any, pointer: str | None) -> Any:
    if not pointer or pointer == "/":
        return document
    if not pointer.startswith("/"):
        raise ProbeError(f"json_pointer must start with '/', got {pointer!r}")
    node = document
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(node, list):
            try:
                node = node[int(token)]
            except (ValueError, IndexError) as exc:
                raise ProbeError(f"json_pointer index {token!r} not found") from exc
        elif isinstance(node, Mapping):
            if token not in node:
                raise ProbeError(f"json_pointer key {token!r} not found")
            node = node[token]
        else:
            raise ProbeError(f"json_pointer cannot descend through {type(node).__name__}")
    return node


def _load_record(raw: Mapping[str, Any], base_dir: Path) -> dict[str, Any]:
    """Resolve an inline evidence record or an evidence record in a JSON artifact."""
    merged: dict[str, Any] = {}
    artifact = raw.get("artifact")
    if artifact:
        path = (base_dir / str(artifact)).resolve()
        if path.suffix.lower() != ".json":
            raise ProbeError(f"artifact must be JSON: {artifact}")
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ProbeError(f"artifact not found: {artifact}") from exc
        except json.JSONDecodeError as exc:
            raise ProbeError(f"invalid JSON artifact {artifact}: {exc}") from exc
        selected = _json_pointer(document, raw.get("json_pointer"))
        if not isinstance(selected, Mapping):
            raise ProbeError(f"artifact selection must be an object: {artifact}")
        merged.update(selected)
    merged.update({k: v for k, v in raw.items() if k not in {"artifact", "json_pointer"}})
    if artifact:
        merged["artifact"] = str(artifact)
        if raw.get("json_pointer"):
            merged["json_pointer"] = raw["json_pointer"]
    return merged


def _first(mapping: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


@dataclass(frozen=True)
class Evidence:
    kind: str
    status: str
    capacity_usd: float | None
    capacity_lcb_usd: float | None
    certified: bool
    unit: str
    window: str | None
    sample_count: int | None
    coverage: float | None
    mode: str | None
    artifact: str | None
    raw: Mapping[str, Any]

    @classmethod
    def from_record(cls, kind: str, record: Mapping[str, Any]) -> "Evidence":
        status = _status(_first(record, ("status", "verdict", "gate", "result")))
        capacity = _finite_nonnegative(
            _first(
                record,
                (
                    "capacity_usd",
                    "capacity_notional_usd",
                    "max_notional_usd",
                    "portfolio_capacity_usd",
                ),
            ),
            field=f"{kind}.capacity_usd",
        )
        lcb = _finite_nonnegative(
            _first(
                record,
                (
                    "capacity_lcb_usd",
                    "capacity_usd_lcb",
                    "certified_capacity_usd",
                    "capacity_lower_bound_usd",
                ),
            ),
            field=f"{kind}.capacity_lcb_usd",
        )
        sample_raw = _first(record, ("sample_count", "n", "observations"))
        sample_count = None if sample_raw is None else int(sample_raw)
        if sample_count is not None and sample_count < 0:
            raise ProbeError(f"{kind}.sample_count must be non-negative")
        coverage = _finite_nonnegative(
            _first(record, ("coverage", "coverage_ratio", "snapshot_coverage")),
            field=f"{kind}.coverage",
        )
        if coverage is not None and coverage > 1:
            raise ProbeError(f"{kind}.coverage must be in [0, 1]")
        return cls(
            kind=kind,
            status=status,
            capacity_usd=capacity,
            capacity_lcb_usd=lcb,
            certified=bool(record.get("certified", False)),
            unit=str(record.get("unit", "USD")).strip().upper(),
            window=None if record.get("window") is None else str(record.get("window")),
            sample_count=sample_count,
            coverage=coverage,
            mode=None if record.get("mode") is None else str(record.get("mode")).lower(),
            artifact=None if record.get("artifact") is None else str(record.get("artifact")),
            raw=record,
        )

    def proxy_capacity(self) -> float | None:
        if self.status != PASS or self.unit != "USD":
            return None
        return self.capacity_lcb_usd if self.capacity_lcb_usd is not None else self.capacity_usd


def _evidence_blockers(
    evidence: Evidence | None,
    *,
    sleeve: str,
    leg: str,
    requirement: Mapping[str, Any],
    expected_window: str | None,
) -> list[dict[str, Any]]:
    prefix = {"sleeve": sleeve, "leg": leg, "evidence": None if evidence is None else evidence.kind}
    if evidence is None:
        return [{**prefix, "code": "MISSING_EVIDENCE", "detail": "required evidence record is absent"}]
    blockers: list[dict[str, Any]] = []
    if evidence.status != PASS:
        # A missing/failed record is already a complete fail-closed explanation.
        # Do not bury the binding cause under secondary missing-field diagnostics.
        return [{**prefix, "code": "EVIDENCE_STATUS_NOT_PASS", "detail": evidence.status}]
    if evidence.unit != "USD":
        blockers.append({**prefix, "code": "INVALID_CAPACITY_UNIT", "detail": evidence.unit})
    if bool(requirement.get("require_certified", True)) and not evidence.certified:
        blockers.append({**prefix, "code": "UNCERTIFIED_EVIDENCE", "detail": "certified=true is required"})
    if bool(requirement.get("require_lcb", True)) and evidence.capacity_lcb_usd is None:
        blockers.append({**prefix, "code": "MISSING_CAPACITY_LCB", "detail": "lower-bound USD capacity is required"})
    min_samples = requirement.get("min_samples")
    if min_samples is not None and (evidence.sample_count is None or evidence.sample_count < int(min_samples)):
        blockers.append(
            {
                **prefix,
                "code": "INSUFFICIENT_SAMPLES",
                "detail": f"{evidence.sample_count!r} < {int(min_samples)}",
            }
        )
    min_coverage = requirement.get("min_coverage")
    if min_coverage is not None and (evidence.coverage is None or evidence.coverage < float(min_coverage)):
        blockers.append(
            {
                **prefix,
                "code": "INSUFFICIENT_COVERAGE",
                "detail": f"{evidence.coverage!r} < {float(min_coverage):.6g}",
            }
        )
    require_window = bool(requirement.get("require_window", evidence.kind == "timed_depth"))
    if require_window:
        if not evidence.window:
            blockers.append({**prefix, "code": "MISSING_SELECTED_WINDOW", "detail": "window is required"})
        elif expected_window and evidence.window != expected_window:
            blockers.append(
                {
                    **prefix,
                    "code": "SELECTED_WINDOW_MISMATCH",
                    "detail": f"{evidence.window!r} != {expected_window!r}",
                }
            )
    allowed_modes = requirement.get("allowed_modes")
    if allowed_modes is not None:
        allowed = {str(x).lower() for x in allowed_modes}
        if evidence.mode is None:
            blockers.append({**prefix, "code": "MISSING_EVIDENCE_MODE", "detail": sorted(allowed)})
        elif evidence.mode not in allowed:
            blockers.append(
                {
                    **prefix,
                    "code": "EVIDENCE_MODE_NOT_ALLOWED",
                    "detail": f"{evidence.mode!r} not in {sorted(allowed)!r}",
                }
            )
    return blockers


def _normalise_manifest(manifest: Mapping[str, Any], base_dir: Path) -> dict[str, Any]:
    data = copy.deepcopy(dict(manifest))
    portfolio = data.get("portfolio")
    if not isinstance(portfolio, Mapping):
        raise ProbeError("manifest.portfolio must be an object")
    sleeves = data.get("sleeves")
    if not isinstance(sleeves, Sequence) or isinstance(sleeves, (str, bytes)) or not sleeves:
        raise ProbeError("manifest.sleeves must be a non-empty array")

    normalised_sleeves: list[dict[str, Any]] = []
    for sleeve_raw in sleeves:
        if not isinstance(sleeve_raw, Mapping):
            raise ProbeError("each sleeve must be an object")
        sleeve = dict(sleeve_raw)
        name = str(sleeve.get("name", "")).strip()
        if not name:
            raise ProbeError("each sleeve needs a non-empty name")
        weight = _finite_nonnegative(sleeve.get("weight"), field=f"{name}.weight")
        if weight is None or weight <= 0:
            raise ProbeError(f"{name}.weight must be > 0")
        sleeve["name"] = name
        sleeve["weight"] = weight
        legs_raw = sleeve.get("legs")
        if legs_raw is None:
            legs_raw = [
                {
                    "name": name,
                    "hedge_multiplier": 1.0,
                    "evidence": sleeve.get("evidence", {}),
                }
            ]
        if not isinstance(legs_raw, Sequence) or isinstance(legs_raw, (str, bytes)) or not legs_raw:
            raise ProbeError(f"{name}.legs must be a non-empty array")
        legs: list[dict[str, Any]] = []
        for leg_raw in legs_raw:
            if not isinstance(leg_raw, Mapping):
                raise ProbeError(f"{name}: each leg must be an object")
            leg = dict(leg_raw)
            leg_name = str(leg.get("name", "")).strip()
            if not leg_name:
                raise ProbeError(f"{name}: each leg needs a non-empty name")
            hedge = _finite_nonnegative(
                leg.get("hedge_multiplier", 1.0),
                field=f"{name}.{leg_name}.hedge_multiplier",
            )
            if hedge is None or hedge <= 0:
                raise ProbeError(f"{name}.{leg_name}.hedge_multiplier must be > 0")
            evidence_raw = leg.get("evidence", {})
            if not isinstance(evidence_raw, Mapping):
                raise ProbeError(f"{name}.{leg_name}.evidence must be an object")
            evidence: dict[str, Evidence] = {}
            for kind, raw in evidence_raw.items():
                if not isinstance(raw, Mapping):
                    raise ProbeError(f"{name}.{leg_name}.{kind} must be an object")
                record = _load_record(raw, base_dir)
                evidence[str(kind)] = Evidence.from_record(str(kind), record)
            legs.append({"name": leg_name, "hedge_multiplier": hedge, "evidence": evidence})
        sleeve["legs"] = legs
        normalised_sleeves.append(sleeve)
    data["sleeves"] = normalised_sleeves
    return data


def _stage_capacity(data: Mapping[str, Any], kind: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for sleeve in data["sleeves"]:
        weight = float(sleeve["weight"])
        for leg in sleeve["legs"]:
            ev: Evidence | None = leg["evidence"].get(kind)
            cap = None if ev is None else ev.proxy_capacity()
            hedge = float(leg["hedge_multiplier"])
            sleeve_cap = None if cap is None else cap / hedge
            portfolio_cap = None if sleeve_cap is None else sleeve_cap / weight
            rows.append(
                {
                    "sleeve": sleeve["name"],
                    "leg": leg["name"],
                    "evidence": kind,
                    "status": MISSING if ev is None else ev.status,
                    "certified": False if ev is None else ev.certified,
                    "capacity_usd": None if ev is None else ev.capacity_usd,
                    "capacity_lcb_usd": None if ev is None else ev.capacity_lcb_usd,
                    "hedge_multiplier": hedge,
                    "weight": weight,
                    "sleeve_equivalent_capacity_usd": sleeve_cap,
                    "portfolio_equivalent_capacity_usd": portfolio_cap,
                    "artifact": None if ev is None else ev.artifact,
                }
            )
    usable = [r for r in rows if r["portfolio_equivalent_capacity_usd"] is not None]
    complete = len(usable) == len(rows)
    binding = min(usable, key=lambda r: r["portfolio_equivalent_capacity_usd"]) if usable else None
    return {
        "kind": kind,
        "complete": complete,
        "portfolio_capacity_usd": binding["portfolio_equivalent_capacity_usd"] if complete and binding else None,
        "binding": binding,
        "rows": rows,
    }


def evaluate_manifest(manifest: Mapping[str, Any], *, base_dir: Path | str = ".") -> dict[str, Any]:
    """Evaluate a manifest and return a machine-readable fail-closed gate report."""
    base = Path(base_dir).resolve()
    data = _normalise_manifest(manifest, base)
    portfolio = data["portfolio"]
    target = _finite_nonnegative(
        portfolio.get("target_notional_usd"),
        field="portfolio.target_notional_usd",
    )
    if target is None or target <= 0:
        raise ProbeError("portfolio.target_notional_usd must be > 0")
    safety = _finite_nonnegative(
        portfolio.get("capacity_safety_multiple", 1.0),
        field="portfolio.capacity_safety_multiple",
    )
    if safety is None or safety < 1:
        raise ProbeError("portfolio.capacity_safety_multiple must be >= 1")
    required_capacity = target * safety

    weights = [float(s["weight"]) for s in data["sleeves"]]
    weight_target = float(portfolio.get("weight_sum_target", 1.0))
    weight_tolerance = float(portfolio.get("weight_tolerance", 1e-6))
    blockers: list[dict[str, Any]] = []
    if abs(sum(weights) - weight_target) > weight_tolerance:
        blockers.append(
            {
                "sleeve": None,
                "leg": None,
                "evidence": None,
                "code": "INVALID_WEIGHT_SUM",
                "detail": f"sum={sum(weights):.12g}, target={weight_target:.12g}",
            }
        )

    for sleeve in data["sleeves"]:
        status = _status(sleeve.get("research_gate"))
        if status != PASS:
            blockers.append(
                {
                    "sleeve": sleeve["name"],
                    "leg": None,
                    "evidence": None,
                    "code": "RESEARCH_GATE_NOT_PASS",
                    "detail": status,
                }
            )

    gates = data.get("gates", {})
    if not isinstance(gates, Mapping):
        raise ProbeError("manifest.gates must be an object")
    required_gates = portfolio.get("required_gates", ["watch_gate", "risk_gate", "ops_gate"])
    if not isinstance(required_gates, Sequence) or isinstance(required_gates, (str, bytes)):
        raise ProbeError("portfolio.required_gates must be an array")
    gate_results: dict[str, str] = {}
    for gate_name in required_gates:
        raw = gates.get(gate_name)
        if isinstance(raw, Mapping):
            status = _status(_first(raw, ("status", "verdict", "gate", "result")))
        else:
            status = _status(raw)
        gate_results[str(gate_name)] = status
        if status != PASS:
            blockers.append(
                {
                    "sleeve": None,
                    "leg": None,
                    "evidence": None,
                    "code": "REQUIRED_GATE_NOT_PASS",
                    "detail": f"{gate_name}={status}",
                }
            )

    required_evidence = portfolio.get("required_evidence", ["timed_depth", "fill"])
    if (
        not isinstance(required_evidence, Sequence)
        or isinstance(required_evidence, (str, bytes))
        or not required_evidence
    ):
        raise ProbeError("portfolio.required_evidence must be a non-empty array")
    requirements = portfolio.get("evidence_requirements", {})
    if not isinstance(requirements, Mapping):
        raise ProbeError("portfolio.evidence_requirements must be an object")
    selected_window = portfolio.get("selected_window")

    final_rows: list[dict[str, Any]] = []
    for sleeve in data["sleeves"]:
        weight = float(sleeve["weight"])
        for leg in sleeve["legs"]:
            hedge = float(leg["hedge_multiplier"])
            capacities: list[float] = []
            row_blockers: list[dict[str, Any]] = []
            evidence_details: dict[str, Any] = {}
            for kind_raw in required_evidence:
                kind = str(kind_raw)
                ev: Evidence | None = leg["evidence"].get(kind)
                req = requirements.get(kind, {})
                if not isinstance(req, Mapping):
                    raise ProbeError(f"evidence_requirements.{kind} must be an object")
                ev_blockers = _evidence_blockers(
                    ev,
                    sleeve=sleeve["name"],
                    leg=leg["name"],
                    requirement=req,
                    expected_window=None if selected_window is None else str(selected_window),
                )
                row_blockers.extend(ev_blockers)
                evidence_details[kind] = None if ev is None else {
                    "status": ev.status,
                    "certified": ev.certified,
                    "capacity_usd": ev.capacity_usd,
                    "capacity_lcb_usd": ev.capacity_lcb_usd,
                    "window": ev.window,
                    "sample_count": ev.sample_count,
                    "coverage": ev.coverage,
                    "mode": ev.mode,
                    "artifact": ev.artifact,
                }
                if ev is not None and not ev_blockers:
                    cap = (
                        ev.capacity_lcb_usd
                        if bool(req.get("require_lcb", True))
                        else ev.proxy_capacity()
                    )
                    if cap is not None:
                        capacities.append(cap)
            blockers.extend(row_blockers)
            leg_cap = (
                min(capacities)
                if len(capacities) == len(required_evidence) and not row_blockers
                else None
            )
            sleeve_cap = None if leg_cap is None else leg_cap / hedge
            portfolio_cap = None if sleeve_cap is None else sleeve_cap / weight
            final_rows.append(
                {
                    "sleeve": sleeve["name"],
                    "leg": leg["name"],
                    "weight": weight,
                    "hedge_multiplier": hedge,
                    "certified_leg_capacity_usd": leg_cap,
                    "sleeve_equivalent_capacity_usd": sleeve_cap,
                    "portfolio_equivalent_capacity_usd": portfolio_cap,
                    "max_structural_weight_at_required_capacity": (
                        None if sleeve_cap is None else sleeve_cap / required_capacity
                    ),
                    "evidence": evidence_details,
                    "blocker_codes": sorted({b["code"] for b in row_blockers}),
                }
            )

    usable_final = [r for r in final_rows if r["portfolio_equivalent_capacity_usd"] is not None]
    complete_final = len(usable_final) == len(final_rows)
    binding_final = (
        min(usable_final, key=lambda r: r["portfolio_equivalent_capacity_usd"])
        if usable_final
        else None
    )
    certified_capacity = (
        binding_final["portfolio_equivalent_capacity_usd"]
        if complete_final and binding_final
        else None
    )
    if certified_capacity is not None and certified_capacity < required_capacity:
        blockers.append(
            {
                "sleeve": binding_final["sleeve"],
                "leg": binding_final["leg"],
                "evidence": "combined",
                "code": "CAPACITY_BELOW_REQUIRED_MULTIPLE",
                "detail": f"{certified_capacity:.2f} < {required_capacity:.2f}",
            }
        )

    # De-duplicate blockers without hiding distinct sleeves/legs.
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for blocker in blockers:
        key = json.dumps(blocker, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            unique.append(blocker)
    blockers = unique

    deploy_open = (
        certified_capacity is not None
        and certified_capacity >= required_capacity
        and not blockers
    )
    stages = {
        "current_depth_proxy": _stage_capacity(data, "current_depth"),
        "timed_depth_proxy": _stage_capacity(data, "timed_depth"),
    }
    manifest_hash = hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": f"sha256:{manifest_hash}",
        "probe_properties": {
            "network_access": False,
            "backtest_run": False,
            "source_artifacts_mutated": False,
            "fail_closed": True,
        },
        "portfolio": {
            "target_notional_usd": target,
            "capacity_safety_multiple": safety,
            "required_capacity_usd": required_capacity,
            "selected_window": selected_window,
            "weight_sum": sum(weights),
            "weight_sum_target": weight_target,
            "required_evidence": [str(x) for x in required_evidence],
            "required_gates": [str(x) for x in required_gates],
        },
        "gates": gate_results,
        "stage_capacity": {
            name: {
                "complete": stage["complete"],
                "portfolio_capacity_usd": stage["portfolio_capacity_usd"],
                "binding": stage["binding"],
            }
            for name, stage in stages.items()
        },
        "certified_execution": {
            "complete": complete_final,
            "portfolio_capacity_usd": certified_capacity,
            "capacity_multiple": (
                None if certified_capacity is None else certified_capacity / target
            ),
            "binding": binding_final,
            "rows": final_rows,
        },
        "blocker_matrix": blockers,
        "deploy_gate": "OPEN" if deploy_open else "CLOSED",
        "deployable_count": 1 if deploy_open else 0,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    p = report["portfolio"]
    cert = report["certified_execution"]
    stage = report["stage_capacity"]

    def money(value: Any) -> str:
        return "n/a" if value is None else f"${float(value):,.0f}"

    def binding_label(stage_record: Mapping[str, Any]) -> str:
        binding = stage_record.get("binding")
        if not binding:
            return "n/a"
        return f"{binding['sleeve']} / {binding['leg']}"

    lines = [
        "# Structural Execution Gate Report",
        "",
        f"- deploy_gate: **{report['deploy_gate']}**",
        f"- deployable_count: **{report['deployable_count']}**",
        f"- target_notional: {money(p['target_notional_usd'])}",
        (
            f"- required_capacity ({p['capacity_safety_multiple']:.2f}x): "
            f"{money(p['required_capacity_usd'])}"
        ),
        (
            f"- current-depth portfolio proxy: "
            f"{money(stage['current_depth_proxy']['portfolio_capacity_usd'])} "
            f"(binding: {binding_label(stage['current_depth_proxy'])})"
        ),
        (
            f"- timed-depth portfolio proxy: "
            f"{money(stage['timed_depth_proxy']['portfolio_capacity_usd'])} "
            f"(binding: {binding_label(stage['timed_depth_proxy'])})"
        ),
        (
            f"- certified execution capacity: {money(cert['portfolio_capacity_usd'])} "
            f"(binding: {binding_label(cert)})"
        ),
        f"- selected_window: `{p['selected_window']}`",
        f"- manifest: `{report['manifest_sha256']}`",
        "",
        (
            "> Fail-closed semantics: current depth is a proxy only; absent or "
            "uncertified timed-depth/fill evidence never becomes PASS."
        ),
        "",
        "## Certified capacity by structural sleeve/leg",
        "",
        (
            "| sleeve | leg | weight | hedge | certified leg cap | "
            "portfolio-equivalent cap | max weight at required capacity | blockers |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in cert["rows"]:
        lines.append(
            (
                "| {sleeve} | {leg} | {weight:.4f} | {hedge:.4f} | {legcap} | "
                "{portcap} | {maxw} | {blockers} |"
            ).format(
                sleeve=row["sleeve"],
                leg=row["leg"],
                weight=row["weight"],
                hedge=row["hedge_multiplier"],
                legcap=money(row["certified_leg_capacity_usd"]),
                portcap=money(row["portfolio_equivalent_capacity_usd"]),
                maxw=(
                    "n/a"
                    if row["max_structural_weight_at_required_capacity"] is None
                    else f"{row['max_structural_weight_at_required_capacity']:.4f}"
                ),
                blockers=", ".join(row["blocker_codes"]) or "—",
            )
        )
    lines.extend(["", "## Execution blocker matrix", ""])
    if report["blocker_matrix"]:
        lines.extend(
            [
                "| sleeve | leg | evidence | code | detail |",
                "|---|---|---|---|---|",
            ]
        )
        for blocker in report["blocker_matrix"]:
            lines.append(
                f"| {blocker.get('sleeve') or '—'} | {blocker.get('leg') or '—'} | "
                f"{blocker.get('evidence') or '—'} | `{blocker['code']}` | "
                f"{blocker.get('detail', '')} |"
            )
    else:
        lines.append("No blockers.")
    lines.extend(
        [
            "",
            "## Capacity identity",
            "",
            (
                "`portfolio_capacity = min(leg_capacity / "
                "(abs(structural_weight) * abs(hedge_multiplier)))`"
            ),
            "",
            (
                "The certified leg capacity is the minimum lower-confidence-bound "
                "capacity across every required evidence family."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _write(path: str | None, text: str) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="JSON qualification manifest")
    parser.add_argument("--json-out", help="write machine-readable gate report")
    parser.add_argument("--md-out", help="write Markdown gate report")
    parser.add_argument(
        "--require-open",
        action="store_true",
        help="return exit code 3 when deploy_gate remains CLOSED",
    )
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest).resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            raise ProbeError("manifest root must be an object")
        report = evaluate_manifest(manifest, base_dir=manifest_path.parent)
    except (OSError, json.JSONDecodeError, ProbeError) as exc:
        print(f"structural execution probe error: {exc}", file=sys.stderr)
        return 2
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    markdown = render_markdown(report)
    _write(args.json_out, payload)
    _write(args.md_out, markdown)
    if not args.json_out and not args.md_out:
        print(markdown)
    if args.require_open and report["deploy_gate"] != "OPEN":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
