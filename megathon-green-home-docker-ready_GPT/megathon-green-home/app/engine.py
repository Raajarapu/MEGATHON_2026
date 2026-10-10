"""Deterministic scoring, retrofit simulation and carbon/value presentation logic."""
from __future__ import annotations
import os
import math
import pandas as pd
import numpy as np

WEIGHTS = {
    "energy_efficiency": 0.25,
    "energy_intensity": 0.20,
    "insulation": 0.15,
    "heating": 0.15,
    "glazing": 0.10,
    "renewables": 0.10,
    "smart_controls": 0.05,
}
RATING_SCORE = {"A": 100, "B": 85, "C": 70, "D": 55, "E": 40, "F": 25, "G": 10}
CLASSIFICATION = [(80, "Highly efficient"), (65, "Good"), (50, "Moderate"), (30, "Poor"), (0, "Very poor")]
ELECTRICITY_EF = float(os.getenv("ELECTRICITY_KG_CO2E_PER_KWH", "0.207"))
GAS_EF = float(os.getenv("GAS_KG_CO2E_PER_KWH", "0.183"))

RETROFITS = {
    "loft_insulation": {"name": "Loft / roof insulation", "impact": "12–18% of space-heating demand", "priority": 1},
    "cavity_wall_insulation": {"name": "Cavity-wall insulation", "impact": "Property-dependent heating-demand reduction", "priority": 1},
    "draught_proofing": {"name": "Draught proofing", "impact": "Reduces uncontrolled heat loss", "priority": 1},
    "double_glazing": {"name": "Double glazing", "impact": "8–12% of space-heating demand", "priority": 2},
    "smart_thermostat": {"name": "Smart thermostat", "impact": "5–10% of heating energy", "priority": 3},
    "heat_pump": {"name": "Air-source heat pump", "impact": "Can reduce gas while increasing electricity", "priority": 4},
    "solar_panels": {"name": "Solar panels", "impact": "Reduces grid electricity imports", "priority": 5},
}

def number(x, default=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default

def classify(score: int) -> str:
    for threshold, label in CLASSIFICATION:
        if score >= threshold:
            return label
    return "Very poor"

def score_property(p: dict) -> dict:
    """Documented monotonic 0-100 component mappings using fixed brief weights."""
    rating = str(p.get("energy_rating", "D")).upper().strip()
    efficiency = RATING_SCORE.get(rating, RATING_SCORE["D"])
    area = max(number(p.get("floor_area"), 1), 1)
    annual_e = max(number(p.get("annual_electricity_kwh")), 0)
    annual_g = max(number(p.get("annual_gas_kwh")), 0)
    intensity = (annual_e + annual_g) / area
    # Monotonic, documented benchmark mapping: 0 at >=250 kWh/m²; 100 at <=50.
    intensity_score = int(round(np.clip((250.0 - intensity) / 200.0 * 100.0, 0, 100)))
    ins = str(p.get("insulation", "poor")).lower()
    insulation_score = {"excellent": 100, "good": 85, "full": 85, "partial": 60, "unknown": 45, "none": 20, "poor": 20}.get(ins, 45)
    heat = str(p.get("heating", "unknown")).lower()
    heating_score = 95 if "heat_pump" in heat or "heat pump" in heat else 80 if "condensing" in heat else 55 if "gas" in heat else 45 if "electric" in heat else 35
    glazing = str(p.get("glazing", "unknown")).lower()
    glazing_score = 95 if glazing in {"triple", "high_performance"} else 75 if glazing == "double" else 25 if glazing == "single" else 45
    renewables_score = 100 if bool(p.get("solar", False)) else 15
    smart_score = 100 if bool(p.get("smart_controls", False)) else 20
    components = {
        "energy_efficiency": {"label": "Energy efficiency rating", "score": efficiency, "weight": WEIGHTS["energy_efficiency"]},
        "energy_intensity": {"label": "Annual energy intensity", "score": intensity_score, "weight": WEIGHTS["energy_intensity"], "kwh_m2": round(intensity, 2)},
        "insulation": {"label": "Insulation", "score": insulation_score, "weight": WEIGHTS["insulation"]},
        "heating": {"label": "Heating system", "score": heating_score, "weight": WEIGHTS["heating"]},
        "glazing": {"label": "Windows and glazing", "score": glazing_score, "weight": WEIGHTS["glazing"]},
        "renewables": {"label": "Renewable energy", "score": renewables_score, "weight": WEIGHTS["renewables"]},
        "smart_controls": {"label": "Smart energy management", "score": smart_score, "weight": WEIGHTS["smart_controls"]},
    }
    raw = sum(v["score"] * v["weight"] for v in components.values())
    score = int(round(raw))
    for v in components.values():
        v["contribution"] = round(v["score"] * v["weight"], 2)
    return {"score": score, "classification": classify(score), "components": components,
            "annual_energy_intensity_kwh_m2": round(intensity, 2),
            "score_method_note": "Fixed weights from the brief. Sub-score mappings are documented in app/engine.py; the intensity mapping uses a 50–250 kWh/m² benchmark range."}

def feasible_retrofits(p: dict) -> list[dict]:
    options = []
    if bool(p.get("has_loft", True)) and str(p.get("insulation", "poor")).lower() not in {"excellent", "good"}:
        options.append("loft_insulation")
    if bool(p.get("cavity_walls", False)) and not bool(p.get("wall_insulated", False)):
        options.append("cavity_wall_insulation")
    if not bool(p.get("draught_proofed", False)):
        options.append("draught_proofing")
    if str(p.get("glazing", "single")).lower() == "single":
        options.append("double_glazing")
    if not bool(p.get("smart_controls", False)):
        options.append("smart_thermostat")
    if str(p.get("heating", "")).lower() != "heat_pump" and bool(p.get("heat_pump_space", True)):
        options.append("heat_pump")
    if not bool(p.get("solar", False)) and bool(p.get("solar_roof_suitable", True)):
        options.append("solar_panels")
    result=[]
    for rid in sorted(set(options), key=lambda x: RETROFITS[x]["priority"]):
        high = rid in {"loft_insulation", "cavity_wall_insulation", "heat_pump", "solar_panels"}
        reason = {
            "loft_insulation": "Prioritise reducing heat loss before changing the heating system.",
            "cavity_wall_insulation": "Cavity walls are present and not recorded as insulated.",
            "draught_proofing": "Draught proofing can reduce uncontrolled heat loss.",
            "double_glazing": "Single glazing is recorded; assess building constraints and cost.",
            "smart_thermostat": "Smart controls are not recorded for this property.",
            "heat_pump": "Space-heating suitability is recorded; evaluate electricity increase and gas reduction together.",
            "solar_panels": "Roof suitability is recorded; estimate grid-import reduction rather than underlying demand.",
        }[rid]
        result.append({"id": rid, "name": RETROFITS[rid]["name"], "suitability": "High" if high else "Medium", "impact_reference": RETROFITS[rid]["impact"], "reason": reason, "priority": RETROFITS[rid]["priority"]})
    return result

def simulate_package(p: dict, retrofit_ids: list[str]) -> dict:
    """Deterministic scenario calculation; per-fuel savings can be negative (increased use)."""
    e0 = max(number(p.get("annual_electricity_kwh")), 0.0)
    g0 = max(number(p.get("annual_gas_kwh")), 0.0)
    space = max(number(p.get("space_heat_kwh"), g0 * 0.75), 0.0)
    base_e = max(number(p.get("base_electricity_kwh"), e0 + number(p.get("pv_offset_kwh"))), 0.0)
    other_gas = max(number(p.get("other_gas_kwh"), max(g0 - space, 0)), 0.0)
    e_after, g_after = e0, g0
    space_multiplier = 1.0
    for rid in retrofit_ids:
        if rid in {"loft_insulation", "cavity_wall_insulation", "draught_proofing"}:
            reduction = {"loft_insulation": 0.15, "cavity_wall_insulation": 0.12, "draught_proofing": 0.06}[rid]
            space_multiplier *= (1 - reduction)
        elif rid == "double_glazing":
            space_multiplier *= 0.90
        elif rid == "smart_thermostat":
            space_multiplier *= 0.93
    g_after = other_gas + space * space_multiplier
    if "heat_pump" in retrofit_ids:
        # Explicit simplified assumption: 90% of space heat is moved from gas to electricity at seasonal COP 2.8.
        gas_space_after = max(space * space_multiplier * 0.10, 0.0)
        heat_pump_electricity = (space * space_multiplier * 0.90) / 2.8
        g_after = other_gas + gas_space_after
        e_after += heat_pump_electricity
    if "solar_panels" in retrofit_ids:
        # PV offset capped at current imported electricity; assume 25% import reduction where no system-specific yield is supplied.
        e_after = max(0.0, e_after - min(e_after * 0.25, max(number(p.get("base_electricity_kwh"), e_after) * 0.25, 0)))
    e_saving, g_saving = e0 - e_after, g0 - g_after
    carbon = e_saving * ELECTRICITY_EF + g_saving * GAS_EF
    total0, total1 = e0 + g0, e_after + g_after
    improvement = ((total0 - total1) / total0 * 100) if total0 > 0 else None
    return {"baseline_electricity_kwh": round(e0, 2), "after_electricity_kwh": round(e_after, 2), "electricity_saving_kwh": round(e_saving, 2),
            "baseline_gas_kwh": round(g0, 2), "after_gas_kwh": round(g_after, 2), "gas_saving_kwh": round(g_saving, 2),
            "baseline_purchased_kwh": round(total0, 2), "after_purchased_kwh": round(total1, 2),
            "improvement_potential_pct": round(improvement, 2) if improvement is not None else None,
            "carbon_reduction_kg_co2e": round(carbon, 2), "emissions_factors": {"electricity_kg_co2e_per_kwh": ELECTRICITY_EF, "gas_kg_co2e_per_kwh": GAS_EF},
            "assumptions": ["Scenario impacts are simplified estimates, not guarantees.", "Insulation, glazing and thermostat impacts are applied sequentially to space-heating demand.", "Heat-pump scenario assumes 90% of space heat is transferred from gas to electricity at seasonal COP 2.8.", "Solar scenario assumes up to 25% grid-import reduction when no property-specific PV yield is available."]}
