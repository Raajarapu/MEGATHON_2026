# Architecture and algorithm notes

```text
CSV inputs (properties / monthly energy / sales / reference labels)
                         |
              Schema validation + joins
                         |
        Feature engineering (Python / Pandas)
             /                           \
  Weighted Green Home Score       Supervised ML pipelines
  fixed seven component weights   RF regression baselines
             |                    improvement potential
  Feasibility + retrofit rules    per-fuel savings / value range
             \                           /
                Deterministic scenario engine
                - ordered retrofit interactions
                - electricity and gas separately
                - documented carbon factors
                - estimate package net changes
                           |
                 Flask REST API / Gunicorn
                           |
            Responsive HTML/CSS/JS dashboard
```

## Algorithm selection

- **Green Home Score:** deterministic weighted sum, not ML. Weights are fixed by the brief: 25/20/15/15/10/10/5 percent.
- **Feature engineering:** annual purchased energy per m²; month aggregation/annualisation; property and heating/insulation/glazing/solar/control features; property-level split preservation.
- **Baseline supervised estimators:** RandomForestRegressor pipelines with median imputation, missingness indicators, scaling for numeric features and one-hot encoding for categorical features. One pipeline is trained per reference target. `random_state=42`.
- **Recommendations:** transparent rules for feasibility and ordering. This avoids training a recommender on arbitrary labels; a learned ranking model can be added only when label provenance supports it.
- **Savings what-if:** deterministic scenario simulator. Space-heating impacts compound sequentially; heat-pump electricity increase and gas reduction are both shown; solar changes grid imports rather than underlying demand.
- **Carbon:** `ΔC = Δelectricity × EF_electricity + Δgas × EF_gas`, where each Δ is baseline minus scenario consumption.
- **Property value:** regression baselines estimate supplied low/high/central reference values from property features. Output is indicative association, not causal uplift.
- **Unknown synthetic data:** inference upload validates a minimum schema and accepts optional fields with defaults. Retraining is a separate explicit operation and requires documented reference labels and valid property-level train/validation/test splits.

## Assumptions to validate before judging

The scenario simulator includes illustrative assumptions for retrofit effects and heat-pump COP. These are made visible in the API output. Replace illustrative emissions factors with a cited current source and record its date, region, units and scope. Validate all assumptions against official/credible sources before describing results as real-world estimates.
