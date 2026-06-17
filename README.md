# enhancer-flow

Conditional enhancer DNA design with **categorical flow maps** in
JAX / Flax NNX, validated by a DeepSTARR sequence→activity oracle.

> **Headline result.** A learned two-time **flow map** — trained by Lagrangian
> self-distillation with a JVP tangent condition — generates biologically
> realistic, activity-controllable enhancer sequences in **1–2 neural-function
> evaluations (NFE)**, matching the quality of a 100-step ODE at a fraction of
> the cost. Phases 1–3 establish the foundation and ablations; **Phase 4 is the
> primary contribution**.


---



## References
Key papers:

- Roos et al. 2026 — *Categorical Flow Maps* (the `endpoint` parametrization)
- Stark et al. 2024 — *Dirichlet Flow Matching* ([arXiv:2402.05841](https://arxiv.org/abs/2402.05841))
- de Almeida et al. 2022 — *DeepSTARR* (*Nature Genetics*, [s41588-022-01048-5](https://www.nature.com/articles/s41588-022-01048-5))
- Peebles & Xie, 2023 — Scalable Diffusion Models with Transformers (DiT, arXiv:2212.09748) — the AdaLN-Zero backbone our denoiser adapts.
