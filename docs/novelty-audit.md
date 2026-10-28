# Preliminary novelty audit

This is a scoped initial audit, not a systematic review or novelty guarantee.

| Prior work | Overlap | Consequence for this project |
|---|---|---|
| Zhao et al., ComRisk (2022) | Bankruptcy GNNs combining firm-level risk with heterogenous relations and industry/area hyperedges | Do not claim that GNNs or relational bankruptcy prediction are new. |
| Chen and Fan, RGCT-PreRisk (2025) | Financial relational graphs, cross-temporal/cross-company contrastive pretraining, risk prediction | Compare graph construction and temporal learning directly; avoid broad “first temporal relational GNN” claims. |
| Sun et al., RA-HTGNN (2026) | Relation-specific graph convolution, relation attention, recurrent temporal encoding, one-year distress prediction | Directly overlaps the synopsis architecture. Distinguish the reliability/persistence treatment of feature-derived peer edges and the controlled graph perturbation evaluation, if the full paper review confirms that gap. |

## Candidate contribution (unverified)

Study whether neighborhood persistence provides useful reliability information for evolving feature-similarity graphs in a strictly point-in-time, chronological distress benchmark. Evaluate the mechanism with a simple relation gate, conventional GCN/GraphSAGE, feature-only and temporal baselines, edge perturbations, leave-one-relation-out ablations, and company-clustered bootstrap intervals. The contribution is empirical and methodological only if results support it. No claim of causal contagion is made.

## Gate before paper submission

1. Complete a systematic literature search across temporal GNN, graph robustness, financial distress, and bankruptcy prediction.
2. Verify the target event horizon and filing-time availability in the dataset.
3. Compare the implemented equations and graph construction against the closest prior papers.
4. Revise the contribution statement if the reliability mechanism is not substantively distinct.
5. Select an A* host workshop only after checking its current scope, track, and deadline.
