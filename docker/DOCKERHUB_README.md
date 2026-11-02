# LIRRA

**LIRRA** (Learning-Augmented Interval-STL for Recursively-feasible Runtime Assurance) lets a
vehicle skip most online signal temporal logic synthesis: a learned policy proposes whole
trajectories in about a millisecond, each proposal is verified against the full interval-STL
synthesis constraints, and a cached verified plan with delay-compensated synthesis takes over when a
proposal is rejected. Satisfaction and recursive feasibility hold for any proposer and any solver
completion time.

Paper: *Skip the Solve: Learned Trajectory Proposals for Recursively Feasible Interval-STL Runtime
Assurance* (L4DC 2027, under review)

Source: [github.com/evannsmc/lirra](https://github.com/evannsmc/lirra)

Documentation, results and videos: [evannsmc.github.io/rl-stl](https://evannsmc.github.io/rl-stl/)

## What's Included

This image ships a complete environment for every LIRRA experiment:

- **Base:** `ghcr.io/astral-sh/uv` (Debian bookworm, Python 3.12)
- **Python:** MuJoCo/MJX 3.13.0, MuJoCo Playground 0.2.0, Brax 0.14.2, JAX 0.11.1 (CPU), Flax,
  `stlpy` 0.3.0, `gurobipy` 13.0.3, and the LIRRA code (`rl_stl`)
- **System:** EGL/OSMesa offscreen rendering, ffmpeg
- **Trained proposers** (`policies/`) for the Skydio X2 and the double integrator; the remaining
  X2 and blimp proposers are coming soon

## Quick Start

```bash
git clone https://github.com/evannsmc/lirra.git && cd lirra/docker
make pull             # pulls this image and tags it locally
make run              # starts the container with the repository mounted
```

**You need your own Gurobi licence** for the Skydio X2 and blimp synthesis problems (the bundled
size-limited licence only covers the double integrator). Use a Gurobi Web License Service (WLS)
licence, free for academics, saved as `~/gurobi.lic`; `make run` mounts it. Step-by-step instructions:
[Getting a Gurobi Licence](https://github.com/evannsmc/lirra#getting-a-gurobi-licence).

## Run Examples

```bash
make test                                  # unit tests
make eval SYS=x2 SPEC=theta_plus           # I-STL vs LIRRA vs unchecked proposer
make video SYS=x2 SPEC=phi_plus            # render an episode
make table                                 # results tables
```

See the [full target reference](https://github.com/evannsmc/lirra/blob/main/docker/README.md),
including training from scratch.

## License

MIT
