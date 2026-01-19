## Methodology-TLDR:

>For a detailed description of the methodology, please read [here](/methodology.md).


This analysis computes two different objects, on purpose. Conflating them is the root of the 653m confusion.

**Object A: queue overshoot (wealth-space abstraction).** We reproduce the debate’s headline queue overshoot using a stylized wealth-space queue. This is not a production replay. Its role is to stress-test the distributional shape induced by queue-like prioritization when expressed as equity haircuts. Importantly, the haircut capacity is **PnL-only** rather than equity + principal, because principal/cash is returned when a position is force-closed; counting it as “haircut” would be a numéraire error.  

**Object B: production overshoot vs needed (counterfactual, PnL dollars).** Separately, we measure a production-anchored notion of wealth removed from winners using a two-pass counterfactual on the realized price path: (i) replay with ADL fills applied to tracked winner states, versus (ii) replay where ADL fills do not update winner states, while the realized price path is held fixed. The per-wave difference in end equities defines production wealth removed. 

To compare this to the immediate solvency need, we compute a needed budget proxy per wave from ADL fills as the sum of the mark-to-execution gaps times size. This proxy isolates the instantaneous bankruptcy-gap transfer implied by forced closes, distinct from any post-wave opportunity cost. 

**Wave construction matters.** We cluster ADL fills into global time waves (rather than per-coin waves) because a single solvency episode can span multiple markets; per-coin partitioning can double count the same systemic event. Loser deficits are computed from loser-side equity fields, not from winner-side equity.  

**Identification knob: evaluation horizon.** The counterfactual equity delta can be evaluated at the wave end, or at a horizon after the wave end while updating only the realized price path. Varying this horizon isolates an opportunity cost channel (what forced closing prevented the winner from riding) and makes explicit that strategic participants care about short-horizon gradients, while passive participants care more about point estimates.  

**Why the two headlines diverge.** A large wealth-space equity overshoot does not map 1:1 into PnL dollars for two reasons. First, winners are often overcollateralized: equity can be several times larger than unrealized PnL, so equity-based aggregates can overstate PnL capacity unless the mapping is made explicit. Second, behavior is heterogeneous: many accounts reverse exposure quickly after ADL, so lost future PnL depends strongly on horizon and on an account’s propensity to undo. We quantify this using an undo fraction computed from post-ADL non-ADL fills.  

The upshot is not that one headline is right and the other is wrong. The upshot is that they answer different questions: the wealth-space queue overshoot diagnoses a distributional pathology of queue-like rules, while the two-pass counterfactual provides a production-anchored estimate of the PnL-channel impact under a fixed price-path assumption.
