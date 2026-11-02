# Trained Proposers

`policies/<system>/<spec>.npz` is the trained proposer (behavior cloning, DAgger, margin-weighted
cloning and PPO) behind the paper's results for that system and specification. The `make eval`,
`make video` and `make paper-evals` targets in `docker/` use these files by default.

| System | `SPEC` | Specification | Status |
|---|---|---|---|
| `blimp` | `phi` | φ, square | coming soon |
| `blimp` | `phi_plus` | φ, cube | coming soon |
| `blimp` | `phi_plus_u` | φ, cube, uncertain | coming soon |
| `blimp` | `phi_u` | φ, square, uncertain | coming soon |
| `blimp` | `theta` | θ, square | coming soon |
| `blimp` | `theta_plus` | θ, cube | coming soon |
| `blimp` | `theta_plus_u` | θ, cube, uncertain | coming soon |
| `blimp` | `theta_u` | θ, square, uncertain | coming soon |
| `di` | `phi` | φ, square | included |
| `di` | `phi_plus` | φ, cube | included |
| `di` | `theta` | θ, square | included |
| `di` | `theta_plus` | θ, cube | included |
| `x2` | `phi` | φ, square | included |
| `x2` | `phi_plus` | φ, cube | included |
| `x2` | `phi_plus_u` | φ, cube, uncertain | coming soon |
| `x2` | `phi_u` | φ, square, uncertain | coming soon |
| `x2` | `theta` | θ, square | included |
| `x2` | `theta_plus` | θ, cube | included |
| `x2` | `theta_plus_u` | θ, cube, uncertain | coming soon |
| `x2` | `theta_u` | θ, square, uncertain | coming soon |

Proposers marked *coming soon* will be added shortly; until then, train one with the pipeline in
`docker/README.md` and pass it with `POLICY=...`.
