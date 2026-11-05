# Trained Proposers

`policies/<system>/<spec>.npz` is the trained proposer (behavior cloning, DAgger, margin-weighted
cloning and PPO) behind the paper's results for that system and specification. The `make eval`,
`make video` and `make paper-evals` targets in `docker/` use these files by default.

| System | `SPEC` | Specification | Status |
|---|---|---|---|
| `blimp` | `phi` | φ, square | included |
| `blimp` | `phi_plus` | φ, cube | included |
| `blimp` | `phi_plus_u` | φ, cube, uncertain | included |
| `blimp` | `phi_u` | φ, square, uncertain | included |
| `blimp` | `theta` | θ, square | included |
| `blimp` | `theta_plus` | θ, cube | included |
| `blimp` | `theta_plus_u` | θ, cube, uncertain | included |
| `blimp` | `theta_u` | θ, square, uncertain | included |
| `di` | `phi` | φ, square | included |
| `di` | `phi_plus` | φ, cube | included |
| `di` | `theta` | θ, square | included |
| `di` | `theta_plus` | θ, cube | included |
| `x2` | `phi` | φ, square | included |
| `x2` | `phi_plus` | φ, cube | included |
| `x2` | `phi_plus_u` | φ, cube, uncertain | included |
| `x2` | `phi_u` | φ, square, uncertain | included |
| `x2` | `theta` | θ, square | included |
| `x2` | `theta_plus` | θ, cube | included |
| `x2` | `theta_plus_u` | θ, cube, uncertain | included |
| `x2` | `theta_u` | θ, square, uncertain | included |
