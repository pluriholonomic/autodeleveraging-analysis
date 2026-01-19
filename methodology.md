## Methodology: wealth-space reconstruction of Hyperliquid Oct 10-11 ADL (and the "\$653m" debate)

This document is the single methodology note for the OSS reproduction in this repository.

Primary references (and what we take from each):

- **Addendum clarifications (two numeraires / reconstruction definitions)**: [`Tarun-ADL-paper-Addendums.md`](https://raw.githubusercontent.com/thogiti/WritingExtras/main/Tarun-ADL-paper-Addendums.md)
- **Paper that triggered the controversy**: [`arXiv:2512.01112`](https://arxiv.org/abs/2512.01112)

---

### 0) Why this exists (motivation + controversy)

The controversy (in one sentence) is a **numéraire mismatch**:

- Production ADL mechanisms (used in Hyperliquid, Binance, and others) are implemented in **contract space** (close contracts $q$ according to a queue), while
- The paper's fairness / efficiency discussion (and the public "\$653m" claim) is about **wealth space** (equity haircuts, i.e., dollars of equity lost).


The addendum's point is not subtle: if you do not reconcile these spaces, "haircut dollars" can be meaningless.

Specifically, since users simply get their cash back upon ADL, the cash component of equity (in wealth space) should not be counted as a loss due to ADL.


This repository exists to make that reconciliation explicit and reproducible.

---

### 1) Two spaces, two pro-ratas (core conceptual hygiene)

#### 1.1 Contract space vs wealth space (queue)

Production ADL: choose which positions to close and by how much in **contracts**.

Our analysis: compare mechanisms via a **wealth-space** object that lives in equity dollars:

$$
H_t^{\text{prod}} = \sum_{i \in W(t)}\left(e^{\text{noADL}}_{t,\text{end}}(i) - e^{\text{ADL}}_{t,\text{end}}(i)\right)_+.
$$

This is exactly the addendum's "realized wealth-space haircut" framing (see "Wealth-Space Reconstruction of Hyperliquid ADL" in [`Tarun-ADL-paper-Addendums.md`](https://raw.githubusercontent.com/thogiti/WritingExtras/main/Tarun-ADL-paper-Addendums.md)).

#### 1.2 Pro-rata in equity vs pro-rata in contracts

The addendum explicitly distinguishes:

- **wealth pro-rata**: same fraction of equity lost by every winner (scalar $\lambda$)
- **contracts pro-rata**: losses proportional to contract counts $q_i$

This OSS repo sticks to wealth-space comparisons whenever we are talking about **equity dollars**.

---

### 2) Data sources (what must be on disk)

This reproduction uses two upstream repos:

#### 2.1 HyperReplay (ground-truth event stream)

We use:

- `HyperReplay/data/raw/20_fills.json`, `21_fills.json` (fills, incl. ADL fills + liquidation metadata such as `markPx`)
- `HyperReplay/data/raw/20_misc.json`, `21_misc.json` (funding, deposits/withdrawals/transfers, accountValue overrides)
- clearinghouse snapshot JSONs at `t = 1760126694218` (extracted from the tar archive shipped by HyperReplay)
- `HyperReplay/data/canonical/adl_detailed_analysis_REALTIME.csv` (canonical ADL-fill table used for **winner set** extraction and for **equity/PnL ratios**)

#### 2.2 HyperMultiAssetedADL (canonical ADL tables / liquidation exports)

We use (depending on upstream layout):

- **Combined mode**: a single `.../adl_detailed_analysis_REALTIME.csv` that includes both winners and loser-side rows, separated by `is_negative_equity`
- **Separate mode (fallback)**:
  - `.../adl_detailed_analysis_REALTIME.csv` (winner-side ADL fills)
  - `.../liquidations_full_12min.csv` (legacy liquidation export)

---

### 3) Definitions (objects we compute)

#### 3.1 Waves (global time waves)

We define **global waves** over all coins by gap clustering:

- sort ADL-fill rows by time
- start a new wave when the time gap exceeds `gap_ms` (default 5000ms)
- wave $t$ is the interval $[t_{\text{start}}(t), t_{\text{end}}(t)]$

Implementation: `OSS/src/oss_adl/bad_debt.py::cluster_global_time`.

Why global (not per coin): because a single solvency episode often spans multiple markets; per-coin partitioning can double-count the same systemic event.

#### 3.2 Loser deficit $D_t$ (bad debt)

Addendum-consistent definition:

$$
D_t = \sum_{j \in \text{losers}(t)} (-e_t(j))_+.
$$

Implementation: `OSS/src/oss_adl/bad_debt.py::compute_loser_waves`.

**Crucial implementation detail**: we compute the loser-side equity from the canonical table's `liquidated_*` fields; we do **not** inspect the ADL winner's equity and pretend it is loser equity.

Verification (this matters, and it is not a cosmetic choice): on `HyperReplay/data/canonical/adl_detailed_analysis_REALTIME.csv` for Oct 10-11,

- `liquidated_total_equity < 0` on 71.33% of rows (as expected for loser-side liquidations), but
- winner-side `total_equity < 0` on only 0.86% of rows.

If you incorrectly compute $D_t$ using winner `total_equity` rather than `liquidated_total_equity`, the aggregate deficit changes from \$100.06M to \$16.98M and several waves move by about \$10M. It also changes which waves are detected as having positive deficit. So the loser-side fields must be used if we want $D_t$ to mean "loser deficit".

#### 3.3 Needed budget $B_t^{\text{needed}}$ (bankruptcy-gap proxy)

Per ADL fill $k$ with liquidation mark $p^{mark}_k$, execution price $p^{exec}_k$, size $q_k$:

$$
\text{needed}_k := \left|p^{mark}_k - p^{exec}_k\right|\cdot |q_k|.
$$

Wave aggregate:

$$
B_t^{\text{needed}} := \sum_{k \in \text{ADL fills in wave }t} \text{needed}_k.
$$

Implementation: `OSS/src/oss_adl/two_pass_replay.py::run_two_pass_for_waves` (we parse `markPx` and `px` directly from the raw fill stream).

Interpretation: this is the instantaneous transfer implied by closing contracts across a mark-bankruptcy gap, not the subsequent opportunity cost.

#### 3.4 Production wealth removed $H_t^{\text{prod}}$ (two-pass counterfactual)

We reconstruct the addendum's object using a two-pass replay:

- **ADL-on pass**: apply all fills (including ADL fills) to the tracked winner account states
- **No-ADL pass**: skip applying ADL fills to account states **but still update the realized price path**

Then:

$$
H_t^{\text{prod}} = \sum_{i \in W(t)}\left(e^{\text{noADL}}_{t,\text{end}}(i) - e^{\text{ADL}}_{t,\text{end}}(i)\right)_+.
$$

Implementation: `OSS/src/oss_adl/two_pass_replay.py::run_two_pass_for_waves`.

This is the addendum's "measure actual equity lost vs a no-ADL counterfactual on the realized path" requirement.

#### 3.5 Overshoot vs needed (PnL-\$ headline)

Per wave:

$$
\text{overshoot}^{\text{prod}}(t) := H_t^{\text{prod}} - B_t^{\text{needed}}.
$$

We report the end-of-wave headline as the sum over waves at horizon $\Delta=0$ms:

$$
O(0) := \sum_t \left(H_t^{\text{prod}}(\Delta=0) - B_t^{\text{needed}}\right).
$$

Implementation:

- per-horizon totals: `OSS/out/eval_horizon_sweep_gap_ms=5000.csv` (produced by `oss-adl replay`)
- robustness summary: `OSS/out/overshoot_robustness.json` (discounted overshoot + horizon band)

#### 3.6 Evaluation horizon $\Delta$ (time-discount / opportunity cost channel)

Holding the wave partition fixed, we optionally evaluate equities at:

$$
t_{\text{eval}} = t_{\text{end}} + \Delta
$$

but **after** $t_{\text{end}}$ we only update the realized price path (no additional account state updates). This isolates price evolution after the wave end as an opportunity-cost component.

Implementation: `OSS/src/oss_adl/two_pass_replay.py` (parameter `eval_horizon_ms`).

#### 3.7 Queue overshoot (equity-\$ headline $\approx 650m$)

Separately from the two-pass replay, we reproduce the debate's "queue overshoot in equity dollars" number using a stylized **wealth-space queue**:

- build per-position winner "score" $s_i$ (here: `closed_pnl`)
- build per-position PnL capacity $cap_i$ (here: `position_unrealized_pnl`, with an account-level cap where available)
- set a score-based budget target $B := \min(\sum_i s_i, \sum_i cap_i)$
- allocate haircuts greedily by score until budget exhausted
- compare realized budget to a loser-side deficit proxy per shock cluster and aggregate overshoot

Implementation: `OSS/src/oss_adl/queue_overshoot.py`.

This is a wealth-space abstraction (by design); see the contract-vs-wealth queue discussion in [`Tarun-ADL-paper-Addendums.md`](https://raw.githubusercontent.com/thogiti/WritingExtras/main/Tarun-ADL-paper-Addendums.md).

**Deficit mode (important)**:

- In "combined" mode we default to an equity-based deficit proxy by mapping loser rows' `total_equity` into the deficit term.
- You can change the queue deficit proxy via:
  - `--queue-deficit-select-mode ...` (which loser-side field determines membership in the deficit set)
  - `--queue-deficit-value-mode ...` (which loser-side field determines the deficit magnitude)

#### 3.8 Winner overcollateralization (equity / PnL)

For each winner $u$, define the following quantities at the winner's **first appearance** in the canonical table:

- $E(u)$: the winner's positive equity (from `total_equity`)
- $U(u)$: the winner's positive unrealized PnL (from `total_unrealized_pnl`)

$$
R(u) := \frac{E(u)}{U(u)}.
$$

This is a unitless "equity per dollar of unrealized PnL" ratio.

We report a **trimmed mean** (default trim 3% per tail) to control heavy tails, and we also keep the full set of distribution summaries (mean, median, p10, p90, and p99) in `out/headlines.json`.

Why trimming is justified here (and why we still show the tails): $R(u)$ is heavy-tailed because the denominator `total_unrealized_pnl` can be arbitrarily close to zero, producing extreme ratios that dominate the untrimmed mean. In this dataset the untrimmed mean is 303x, while the median is 4.41x and p99 is 106x (the maximum can be much larger). For the purpose of the naive mapping from an equity-\$ headline into an expected PnL-\$ headline, we want a stable "typical" ratio rather than a mean dominated by near-zero denominators.

Sensitivity: the trimmed mean remains in the same range under reasonable trims. On this dataset:

- 1% trim: 7.61x
- 3% trim (default): 6.66x
- 5% trim: 6.25x
- 10% trim: 5.73x

The tails are not discarded, they are shown explicitly in the histogram (Figure 03) and in the quantiles written to `out/headlines.json`.

Implementation: `oss-adl headlines` (and underlying code in `OSS/src/oss_adl/cli.py`).

#### 3.9 Undo fraction (two-time behavior)

For an ADL fill with signed delta $\Delta_{\text{ADL}}$, scan subsequent non-ADL fills in a window $W$ for the same `(user, coin)`.

The primary "opposite-volume" undo fraction:

$$
\mathrm{undo}_{\mathrm{frac}} = \frac{\min\left(|\Delta_{\text{ADL}}|,\sum_{i:\Delta_i \cdot \Delta_{\text{ADL}} < 0} |\Delta_i|\right)}{|\Delta_{\text{ADL}}|}.
$$

Implementation: `OSS/src/oss_adl/two_time_behavior.py::compute_undo_fraction`.

We also include a net-based variant for robustness:

$$
\mathrm{undo}_{\mathrm{net}} = \frac{\min\left(|\Delta|,\max(0,-\mathrm{sign}(\Delta)\cdot \Delta_{\text{net}})\right)}{|\Delta|}.
$$

Implementation: `OSS/src/oss_adl/two_time_behavior.py::undo_net_fraction`.

---

### 4) Reproduction: commands + artifacts (everything is file-backed)

#### 4.1 One-command end-to-end repro

From `OSS/`:

```bash
./run-all.sh --full
```

This runs the full pipeline from raw HyperReplay data (clones deps, regenerates the canonical CSV, and calls `oss-adl all` with the pinned parameters: `gap-ms 5000`, `gap-seconds 5.0`, `follower-decay-beta 8.0`, `queue-deficit-select-mode total_equity`, `queue-deficit-value-mode total_equity`, `horizons-ms 0,500,1000,2000,5000`, `horizon-ms 0`, `trim-alpha 0.03`), then builds plots and the paper. For a quick figure refresh using the committed `out/` data, run `./run-all.sh` (no `--full`).

#### 4.1.1 Optional robustness sweeps

If you want to stress-test sensitivity:

- **Wave partition sweep** (changes segmentation):

```bash
uv run oss-adl gap-sweep --gaps-ms 250,500,1000,2000,5000,10000
```

- **Evaluation-horizon sweep** (fixed segmentation; changes $\Delta$):
  rerun `oss-adl replay` with a denser `--horizons-ms` grid.

#### 4.2 Output map (what to inspect)

All outputs land under `OSS/out/`:

- **Headline JSON**: `out/headlines.json`
- **Queue overshoot table**: `out/summary_totals.csv`
- **Two-pass horizon sweep totals**: `out/eval_horizon_sweep_gap_ms=5000.csv`
- **Two-pass per-wave outputs (per horizon)**:
  - `out/eval_horizon_sweep/gap_ms=5000/horizon_ms=0/two_pass_equity_delta_by_wave.csv`
  - `out/eval_horizon_sweep/gap_ms=5000/horizon_ms=0/two_pass_wave_prod_haircuts.csv`
  - and similarly for `horizon_ms=5000`
- **Two-time report**: `out/two_time/report.md`
- **Figures (generated)**: `out/figures/*.png`
- **Figures (committed for GitHub rendering)**: `figures/*.png`
- **Policy-per-wave metrics (generated)**: `out/policy_per_wave_metrics.csv` (written by `oss-adl plots`; depends on the two-pass per-wave outputs at `horizon_ms=0`)

#### 4.3 Code map (every file is used)

- `OSS/src/oss_adl/cli.py`: orchestrates all subcommands
- `OSS/src/oss_adl/paths.py`: resolves upstream data locations (env overridable)
- `OSS/src/oss_adl/queue_overshoot.py`: "queue overshoot (equity-\$)" calculation
- `OSS/src/oss_adl/bad_debt.py`: loser deficit wave construction
- `OSS/src/oss_adl/two_pass_replay.py`: two-pass replay + horizon sweep + robustness summary + gap sweep
- `OSS/src/oss_adl/two_time_behavior.py`: undo fraction + strategic/passive classification + reconciliation report
- `OSS/src/oss_adl/plots.py`: generates figures referenced here
- `OSS/src/oss_adl/policy_per_wave.py`: benchmark policy comparisons (pro-rata, vector-md) using two-pass wave outputs
- `OSS/src/oss_adl/vector_mirror_descent.py`: projection routine used by vector-md benchmark
- `OSS/src/oss_adl/adl_contract_pro_rata.py`: discrete contract rounding utilities used in benchmark plots

---

### 5) Results (what you should see)

After running `./run-all.sh --full` (or using the committed `out/` artifacts), open `OSS/out/headlines.json` and `OSS/out/two_time/report.md`.

The figures referenced below are committed under `OSS/figures/` for GitHub rendering. They are generated from `out/figures/` by `./run-all.sh` (full or quick mode).

#### 5.1 Headline bars

![Headline numbers](figures/01_headlines.png)

Interpretation:

- **queue overshoot (equity-\$)**: the ~650m number from the stylized wealth-space queue overshoot table
- **production overshoot vs needed (PnL-\$)**: the ~45-50m end-of-wave two-pass number
- **naive expected (PnL-\$)**: queue overshoot divided by average equity/PnL ratio

#### 5.2 Overshoot vs horizon (opportunity-cost sensitivity)

![Overshoot vs horizon](figures/02_overshoot_vs_horizon.png)

Interpretation:

- $O(\Delta)$ increases with $\Delta$ when there are within/after-wave price moves; this is the opportunity-cost channel.
- We summarize this sensitivity in `out/overshoot_robustness.json` (discounted overshoot + horizon band).

#### 5.3 Winner overcollateralization (equity / PnL)

![Overcollateralization histogram](figures/03_overcollateralization_hist.png)

Interpretation:

- winners are often **overcollateralized** (equity much larger than PnL), so an equity-\$ number can map to a smaller PnL-\$ number.

#### 5.4 Undo fraction distribution (60s)

![Undo fraction histogram](figures/04_undo_fraction_hist.png)

Interpretation:

- many winners do not undo quickly; a small set does.
- this heterogeneity is what makes single-number equity/PnL mapping arguments fragile.

#### 5.5 Reconciling the "missing \$50-55m PnL" (the point of the two-time analysis)

The three headline pieces (from `out/headlines.json`) imply a naive mapping:

- queue overshoot (equity-\$): $H_Q \approx 650\text{m}$
- average winner overcollateralization: $\bar{R} \approx 6.7\times$

so a back-of-the-envelope expected PnL closed is:

$$
P_{\text{naive}} \approx \frac{H_Q}{\bar{R}} \approx \frac{650\text{m}}{6.7} \approx 97\text{m}.
$$

But the observed end-of-wave production overshoot-vs-needed (PnL-\$) from the two-pass replay is:

$$
O(0) \approx 45\text{m},
$$

so the naive mapping appears to miss roughly $50\text{m}$.

This is exactly why we run the two-time experiments:

- the overshoot and the time-discount component are **not evenly spread across winners**
- they concentrate on the **passive / non-reacting** accounts
- those accounts have materially different equity/PnL ratios than the average winner

Concrete evidence is written by the pipeline to:

- `out/two_time/report.md` (group-level summary tables)
- `out/two_time/strategic_accounts.csv` and `out/two_time/passive_accounts.csv` (who is in each group)

The main claim is not "we have the one true ratio." It is the weaker (and defensible) claim:

> **Using one global equity/PnL ratio to map an equity-\$ overshoot into a PnL-\$ overshoot is not valid when the loss is concentrated on a behavioral subpopulation with systematically different ratios.**

#### 5.6 Production vs benchmark haircutting policies (pro-rata and mirror descent)

The two-pass reconstruction gives an empirical production haircut budget per wave, $H_t^{\text{prod}}$, and an empirical needed budget, $B_t^{\text{needed}}$.

To make "excessive relative to alternatives" concrete, we compare production to simple, fully transparent benchmark allocations that target $B_t^{\text{needed}}$ on each wave:

- **Wealth pro-rata (continuous)**: a capped pro-rata allocation in wealth-space dollars.
- **Vector mirror descent (vector-md)**: a projection-based mirror descent construction that matches the budget exactly while spreading haircuts over the capacity vector.
- **Drift pro-rata (integer contracts)**: contract-space pro-rata with whole-contract rounding, using per-wave gap-per-contract estimates.
- **Pro-rata ILP-ish (integer contracts)**: round the wealth pro-rata targets into whole contracts on flattened (user, coin) pairs.
- **Fixed-point ILP (integer contracts)**: a discrete version of wealth pro-rata that solves for a scaling factor so the post-rounding budget matches $B_t^{\text{needed}}$ as closely as possible.

All of these benchmark allocations use winner capacity/score proxies from the canonical REALTIME table and are implemented in `OSS/src/oss_adl/policy_per_wave.py`. The per-wave table they produce is written to `out/policy_per_wave_metrics.csv`.

Method definitions (high-level):

- **Production (reference)**: $H_t^{\text{prod}}$ is measured from the two-pass replay (Section 3.4). It is not an optimization output.
- **Capacity model (benchmarks)**: for each wave, we build a per-user capacity proxy and cap it by positive equity:

  - $U(u)$: the winner's positive unrealized PnL, taken from `total_unrealized_pnl` within the wave window
  - $E(u)$: the winner's positive equity, taken from `total_equity` within the wave window
  - $c_u := \min(U(u), E(u))$ (effective wealth-space capacity; this is the object the code calls `cap_eff`)

  This is the same basic "PnL is haircutted, cash is protected" capacity proxy used elsewhere in the repo, expressed in wealth-space USD so it is comparable to $H_t^{\text{prod}}$.

- **Wealth pro-rata (continuous)**: choose $\lambda$ such that

  $$
  h_u = \min(\lambda \cdot c_u, c_u),\quad \sum_u h_u = \min(B_t^{\text{needed}}, \sum_u c_u).
  $$

  This is a best-case wealth-space benchmark because it has no contract discreteness.

- **Vector mirror descent (vector-md)**: construct a haircut fraction vector $x \in [0,1]^n$ over users that satisfies the exact budget constraint

  $$
  c^T x = B_t^{\text{needed}}
  $$

  using a projection routine (a mirror descent step with zero gradient). The resulting USD haircuts are $h = x \odot c$. This is a smooth allocator that tends to spread haircuts while matching the budget very closely when total capacity is sufficient.

- **Drift pro-rata (integer contracts)**: a base-space pro-rata that is constrained to whole contracts. We simulate it as:

  1) allocate the wave budget across coins proportional to each coin's contribution to $\sum |\mathrm{mark}-\mathrm{exec}|\cdot q$ (using `gap_usd_x_qty`),
  2) within each coin, close integer contracts pro-rata to winner start exposure ($|q|$),
  3) convert integer contract closes into USD losses using the wave's average gap-per-contract,
  4) cap losses by $c_u$ so the output is a wealth-space haircut vector.

- **Pro-rata ILP-ish (integer contracts)**: start from the continuous wealth pro-rata targets $h_u$ and then round to whole contracts:

  1) split each user's target across their coins by dollar-at-risk weights $|q_{u,coin}| \cdot \mathrm{gapPerContract}_{coin}$,
  2) round user-coin targets into whole-contract closes using a greedy remainder procedure.

  It is "ILP-ish" because it approximates the natural integer program "match continuous targets with whole contracts," but it does not solve an exact ILP.

- **Fixed-point ILP (integer contracts)**: the same discrete rounding problem as ILP-ish, but with an additional fixed-point step so the final discrete haircut sum matches $B_t^{\text{needed}}$ rather than systematically undershooting due to integer rounding.

  Concretely, we compute a per-user contract-achievable capacity

  $$
  c_u^{\text{contract}} := \sum_{coin} |q_{u,coin}|\cdot \mathrm{gapPerContract}_{coin}
  $$

  cap it by $c_u$, then binary search a scalar scale factor applied to the continuous budget so that after rounding to whole contracts the realized haircut sum is as close as possible to $B_t^{\text{needed}}$. The scale used (when defined) is written to the `fixed_point_ilp_scale` column in `out/policy_per_wave_metrics.csv`.

Computational complexity (relative, order-of-growth):

Let:

- $n$: number of winner users in a wave
- $m$: number of winner (user, coin) pairs in a wave
- $k$: number of coins active in a wave
- $T$: number of waves
- $E$: number of fill/misc events in the raw replay window

Then, very roughly:

- **Two-pass replay (`oss-adl replay`)**: $O(E)$ time and $O(n \cdot k)$ state in memory (tracked positions + last prices). This dominates end-to-end runtime.
- **Queue overshoot (`oss-adl queue`)**: $O(N \log N)$ where $N$ is the canonical-table row count (sorting + per-coin clustering) plus $O(\sum_t m_t \log m_t)$ for per-cluster priority allocation (sorting by score/capacity within each cluster).
- **Wealth pro-rata (continuous)**: $O(n)$ in the ideal implementation. Our capped allocator is iterative; worst-case $O(n^2)$, but with these data it behaves close to linear because capacities saturate in a small number of rounds.
- **Vector mirror descent (vector-md)**: $O(n \log(1/\varepsilon))$ because the projection onto $\{x\in[0,1]^n: w^T x = B\}$ is found by a 1D root-find / bisection. In this OSS repo we run a single projection step (not a long optimization loop).
- **Drift pro-rata (integer contracts)**: $O(m)$ to build per-coin positions plus per-coin integer apportionment. The naive “largest remainder” rounding is $O(m \log m)$ if implemented via sorting once.
- **Pro-rata ILP-ish (integer contracts)**: rounding continuous USD targets into whole contracts. In our implementation the greedy remainder loop does repeated argmax over candidates, so worst-case $O(m^2)$ (it is not a solver, but it is also not asymptotically optimal).
- **Fixed-point ILP (integer contracts)**: the ILP-ish rounding wrapped in a scalar binary search over the “scale” parameter, so roughly $O(\log S)\times$ (ILP-ish cost), where $S$ is the scale bracket range. In this repo the bisection iteration count is fixed and small (default 18).

Context: an exact mixed-integer program for whole-contract closes is NP-hard in general. The OSS benchmarks are intentionally solver-free and sized so they are reproducible on a laptop.

Reproduction: the policy-per-wave metrics and Figures 05-06 are generated by `oss-adl plots` (which is run automatically by `./run-all.sh` and `oss-adl all`). If you want to run just this step after you already have the two-pass outputs, run:

```bash
uv run oss-adl replay --gap-ms 5000 --horizons-ms 0,500,1000,2000,5000
uv run oss-adl plots --gap-ms 5000
```

![Production vs benchmark policies (per wave)](figures/05_policy_per_wave_performance.png)

Interpretation:

- By construction, benchmark policies that target $B_t^{\text{needed}}$ have near-zero overshoot vs needed (up to caps/rounding). Production does not.
- The "max % haircut" panel shows that production can be materially more concentrated than benchmark allocations even when they raise the same solvency budget.

The cumulative overshoot view makes the headline point crisp:

![Cumulative overshoot vs needed (benchmarks)](figures/06_policy_per_wave_cumulative_overshoot.png)

---

### 6) Skeptic checklist: what this answers (and what it does not)

#### 6.1 "Is \$653m just notional closed?"

No. In the addendum's framing, the relevant quantity is **equity dollars lost** by winners, not notional.
This OSS repo keeps that distinction explicit: wealth-space quantities are always USD equity deltas or USD haircuts.

See [`Tarun-ADL-paper-Addendums.md`](https://raw.githubusercontent.com/thogiti/WritingExtras/main/Tarun-ADL-paper-Addendums.md).

#### 6.2 "Are you double-counting across coins?"

The production two-pass pipeline uses **global waves** across all coins (gap clustering on time), precisely to avoid per-coin double-counting.
The queue-overshoot pipeline is per-coin because it is reproducing a different stylized calculation based on HyperMultiAssetedADL exports.

#### 6.3 "Is the no-ADL counterfactual identifiable?"

Not perfectly. We make one explicit counterfactual choice:

- hold the realized **price path** (as last observed fill prices) fixed,
- remove the ADL state transformation.

This is the addendum's recommended reconstruction object; it is also the main limitation (real markets would have feedback).

#### 6.4 "What should I compare to the paper?"

This OSS reproduction does not attempt to re-argue the full theoretical paper; it focuses on the empirical objects that were debated publicly:

- a wealth-space interpretation of production impacts (two-pass)
- a clear separation of wealth-space vs contract-space interpretations
- robust, file-backed reconciliation of apparently inconsistent headline numbers

The relevant citation to ground the controversy is the paper itself: [`arXiv:2512.01112`](https://arxiv.org/abs/2512.01112).

#### 6.5 Explicit implementation assumptions (code-backed)

This OSS repo is intentionally minimal. The core objects are defined in sections 3.1-3.9; the list below documents additional implementation choices that are easy to miss if you only read the math.

- **Event-time window (two-pass replay)**: we only process events in the window `[SNAPSHOT_TIME_MS, ADL_END_TIME_MS]` with `SNAPSHOT_TIME_MS = 1760126694218` and `ADL_END_TIME_MS = 1760131620000` (`OSS/src/oss_adl/two_pass_replay.py`). This avoids accidental drift from unrelated activity outside the Oct 10-11 episode.
- **Market filtering**: we drop markets with `coin.startswith("@")` and `coin == "PURR/USDC"` in both the two-pass replay and the two-time analysis (`OSS/src/oss_adl/two_pass_replay.py`, `OSS/src/oss_adl/two_time_behavior.py`). This removes internal/synthetic instruments that are not part of the perp ADL mechanism being analyzed.
- **Cash-only baseline (two-pass replay)**: we convert the snapshot into a "cash-only baseline" by subtracting snapshot unrealized PnL from `account_value` before replay (`OSS/src/oss_adl/two_pass_replay.py::load_baseline_states`). This aligns with the "PnL is what gets haircutted" framing used in the addendum-style reconstructions; it does not affect $B_t^{\text{needed}}$, which is computed directly from ADL fills.
- **Equity reconstruction (two-pass replay)**: for tracked users we compute
  - `account_value(t)` by replaying `closedPnl` and `fee` from fills and applying funding/deposit/withdrawal/transfer/account-value-override events, and
  - `unrealized_pnl(t)` by marking perp positions to the last observed fill price per coin (`OSS/src/oss_adl/two_pass_replay.py::compute_unrealized_pnl_for_state`).

  We do not attempt to fully reconstruct venue-internal position accounting (for example, re-averaging entry prices on partial position changes, or marking spot balances). The headline objects are computed as differences between an ADL-on pass and a no-ADL pass under the same reconstruction, which makes many state-approximation errors cancel in $H_t^{\text{prod}}$.
- **Loser deficit aggregation within a wave**: when computing $D_t$, for each wave and each `liquidated_user` we take the minimum observed `liquidated_total_equity` within the wave and then sum the negative minima (`OSS/src/oss_adl/bad_debt.py`). This avoids double-counting the same liquidated account across multiple ADL-fill rows.
- **Queue overshoot unit + caps**: the queue abstraction allocates haircut capacity per `(user, coin)` position identifier (not per account), caps each position by `min(position_unrealized_pnl, total_unrealized_pnl)` when account-level PnL is available, and enforces `cap >= score` as a compatibility quirk with the legacy score-based budget (`OSS/src/oss_adl/queue_overshoot.py`).
- **Follower participation/churn (queue overshoot)**: the queue overshoot reproduction includes a per-coin follower participation factor that decays as `mass <- mass * exp(-beta * haircut/cap)` after each cluster, tracked independently per coin (`OSS/src/oss_adl/queue_overshoot.py`). The pinned reproduction uses `beta = 8.0` (see `OSS/run-all.sh`).
- **Two-time thresholds (strategic vs passive)**: the strategic/passive split is defined by explicit thresholds on fixed windows (default windows: 5s, 60s, 300s; defaults: `min_adl_events=5`, `strategic_min_share_traded_60s=0.5`, `strategic_min_undo_frac_60s=0.25`, `passive_max_share_traded_300s=0.05`) (`OSS/src/oss_adl/two_time_behavior.py`). The report prints these parameters; they can be changed to stress-test the behavioral claims.
- **Plotting-only clipping**: the overcollateralization histogram is clipped at 50x for readability (`OSS/src/oss_adl/plots.py`). The statistics written to `out/headlines.json` are computed on the unclipped distribution (with trimming as described above).
- **Robustness summary defaults**: `out/overshoot_robustness.json` uses default half-lives `[500, 1000, 2000, 5000]` ms and a default horizon band max of `10000` ms unless overridden (`OSS/src/oss_adl/cli.py`).
