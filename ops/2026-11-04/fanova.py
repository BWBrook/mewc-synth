"""fANOVA importance of the eight recipe dimensions for one model's 40 stage-1 trials (PLAN 2026-11-03,
diagnostic only), with optuna's evaluator on seed-1 macro-F1. Trials without a score are left out.
Usage: uv run --with optuna python ops/2026-11-04/fanova.py TRIALS_CSV REGISTRY_JSONL MODEL"""
import json
import sys

import optuna
import pandas as pd
from optuna.distributions import FloatDistribution, IntDistribution

DIMS = {"lr": FloatDistribution(2e-5, 6e-4, log=True), "weight_decay": FloatDistribution(1e-5, 0.1, log=True),
        "blocks": None, "min_lr_frac": FloatDistribution(0.02, 1, log=True),
        "dropout_max": FloatDistribution(0.05, 0.6), "magnitude_max": FloatDistribution(0.2, 1.0),
        "n_aug": IntDistribution(1, 4), "focal_gamma": FloatDistribution(0, 3)}


def main(trials, registry, model):
    t = pd.read_csv(trials)
    t = t[t.model == model]
    f1 = {r["name"]: r["macro_f1"] for r in map(json.loads, open(registry))}
    t = t.assign(f1=t.run.map(f1)).dropna(subset=["f1"])
    dims = {**DIMS, "blocks": IntDistribution(int(t.blocks.min()), int(t.blocks.max()))}
    study = optuna.create_study(direction="maximize")
    for _, r in t.iterrows():
        study.add_trial(optuna.trial.create_trial(params={k: (int(r[k]) if isinstance(d, IntDistribution) else float(r[k]))
                                                          for k, d in dims.items()}, distributions=dims, value=r.f1))
    imp = optuna.importance.get_param_importances(study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=0))
    print(model, len(t), "trials:", {k: round(v, 3) for k, v in imp.items()})
    print(t[list(DIMS) + ["f1"]].corr(method="spearman")["f1"].drop("f1").round(2).to_dict())


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main(*sys.argv[1:4])
