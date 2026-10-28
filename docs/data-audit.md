# Initial data audit

**Source:** `sowide/bankruptcy_dataset`, file `american_bankruptcy_dataset.csv`, downloaded on 2026-10-22 from upstream commit `8bcd8db3b432b1fa6d65e26753b5c9fc6567438d`. The upstream repository includes a CC BY 4.0 license; this project does not redistribute the raw file.

- Upstream Git blob: `6951cb5f1030e9ef570177255afc0064e2ba10af`
- Local SHA-256: `d64ed85ae786d75113455ae238feb24015d19716c10ec1de3a90a52eddc7b04a`

| Check | Observed |
|---|---:|
| Rows | 78,682 |
| Companies | 8,971 |
| Fiscal years | 1999–2018 |
| Numeric feature columns | 18 (`X1`–`X18`) |
| Duplicate company-years | 0 |
| Missing values | 0 |
| `failed` rows | 5,220 |
| Firms with `failed` label | 609 |
| Firms whose label changes across years | 0 |
| Broad industry fields | `Division` (10 values), `MajorGroup` (73 values) |
| Firm count in original publication | 8,262 (CSV currently reports 8,971) |

## Decision

The target is constant within every company: each failed firm is labelled failed for every year in which it appears, rather than only for a documented event horizon. The original paper describes a next-year target, but the downloadable CSV does not match that target behavior. The file also reports 8,971 firms, while the paper reports 8,262. Consequently, this release cannot be treated as a point-in-time next-year target without recovering event dates from a separate, defensible source and linking them to the anonymized IDs. The training command therefore fails closed on this candidate. No model scores or paper results are produced from it.

## Next data requirement

Supply or construct a point-in-time panel with `company_id`, `fiscal_year`, `available_date`, numeric financial features, industry identifiers where licensed, and an outcome event date or explicitly horizon-aligned label. Negative examples must have a complete follow-up window. The loader accepts a prepared `panel.csv`; its target must vary across time when events occur. Before claiming the research question is tested, document event definitions, filing-time availability, right-censoring, and the label construction procedure.
