## Strategic vs passive winners (behavioral classification)
- **Fills loaded**: 5,641,234
- **ADL fills (winner-side)**: 35,022
- **Per-user reaction rows**: 19,351

### Classification thresholds
- **min ADL events**: 5
- **strategic**: share_any_nonadl_fill_60s >= 0.50 and mean_undo_frac_60s >= 0.25
- **passive**: share_any_nonadl_fill_300s <= 0.05

### Headline reconciliation (sanity)
- **queue overshoot (equity-$)**: 653.55M
- **production overshoot-vs-needed (PnL-$, horizon=0ms)**: 45.03M
- **implied equity/PnL ratio to reconcile** (queue/prod): 14.51×

### Overcollateralization (equity / PnL)
- all winners (trimmed mean 3%): **6.66×**

### Overcollateralization (equity / PnL) by group
| group | n users | median | trimmed mean (3%) |
|---|---:|---:|---:|
| strategic | 27 | 2.09× | 3.26× |
| passive | 600 | 10.53× | 17.68× |

### Group comparison (time-discount component via two-pass)
| group | n users | sum Δhaircut (h1-h0) | mean Δhaircut | median Δhaircut |
|---|---:|---:|---:|---:|
| strategic | 30 | 49429.10 | 1647.64 | 10.14 |
| passive | 754 | 4535573.99 | 6015.35 | 35.98 |

### Top accounts
- **strategic**: see `strategic_accounts.csv`
- **passive**: see `passive_accounts.csv`
