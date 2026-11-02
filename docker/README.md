# Docker Setup

Every LIRRA experiment runs inside a Docker container with all dependencies installed. Run the
`make` targets below from this `docker/` directory.

## Dependencies

**Base image:** `ghcr.io/astral-sh/uv` (Debian bookworm, Python 3.12)

**Python environment** (`/opt/lirra-venv`, from `uv.lock`): MuJoCo/MJX 3.13.0, MuJoCo Playground
0.2.0, Brax 0.14.2, JAX 0.11.1 (CPU; CUDA 12 with `make build-gpu`), Flax, Optax, `stlpy` 0.3.0,
`gurobipy` 13.0.3, and the `rl_stl` package from `src/` (editable).

**System libraries:** EGL/OSMesa for offscreen MuJoCo rendering, `ffmpeg` for videos.

## Getting Started

### Option A: Pre-built Image (Fastest)

```bash
make pull        # pulls evannsmc/lirra:latest and tags it locally
make run         # starts the container with the repository mounted at /workspace/lirra
```

### Option B: Build from Source

```bash
make build       # CPU image
make build-gpu   # CUDA 12 image (lirra:cuda) for training on an NVIDIA GPU
make run         # add GPU=1 (and IMAGE_NAME=lirra:cuda) to use the GPU
```

### Gurobi Licence

Synthesis for the Skydio X2 and the blimp needs a full Gurobi licence; the free pip licence only
covers the small double-integrator problems. Inside a container, use a **Web License Service
(WLS)** licence, which is free for academics (machine-locked named-user licences do not work in
containers). Save it as `~/gurobi.lic` (or pass `GUROBI_LIC=/path/to/gurobi.lic`); `make run`
mounts it read-only. The container prints whether it found the licence when you attach.

## Container Lifecycle

| Command | Description |
|---|---|
| `make pull` | Pull the pre-built image from Docker Hub |
| `make push` | Push the local image to Docker Hub (maintainer only) |
| `make build` / `make build-gpu` | Build the CPU / CUDA 12 image from source |
| `make run` | Start the container (headless; rendering is offscreen) |
| `make start` | Restart a stopped container |
| `make stop` | Stop the running container |
| `make kill` | Force-stop the container |
| `make attach` | Open a shell in the running container |

If the container name is already in use: `make stop && make run`.

## Running Experiments

Results are written under `runs/` of the mounted repository, so they persist after the container
stops. `SYS` is `x2`, `blimp` or `di`; `SPEC` selects one of the eight specifications:

| `SPEC` | Specification |
|---|---|
| `theta`, `phi` | θ (leave the keep-out region within 4 s) and φ (leave within 3 s and stay out 1 s), planar square |
| `theta_plus`, `phi_plus` | the same on the 3-D keep-out cube (the paper's θ and φ) |
| `theta_u`, `phi_u` | square with uncertain boundaries |
| `theta_plus_u`, `phi_plus_u` | cube with uncertain boundaries (the paper's θ⁺ and φ⁺) |

### Evaluate the Trained Proposers

`policies/<SYS>/<SPEC>.npz` holds the trained proposer behind the paper's results.

```bash
make test                                  # unit tests
make eval SYS=x2 SPEC=theta_plus           # always-on I-STL, LIRRA, unchecked proposer: 10 starts x 3 levels
make eval-nominal SYS=x2 SPEC=theta_plus_u # uncertain boundaries: synthesis with the nominal boundaries instead
make video SYS=x2 SPEC=phi_plus LEVEL=2    # render one episode: runs/patrol_videos/
make table                                 # results tables from your evaluations: docs/results/main_table.md
```

`make paper-evals` runs every evaluation behind the paper's table (both vehicles, four
specifications) and then `make table`. On a CPU this takes several hours, mostly in the always-on
synthesis baseline.

| Parameter | Default | Meaning |
|---|---|---|
| `SYS` | `x2` | `di`, `x2` or `blimp` |
| `SPEC` | `theta_plus` | one of the eight specifications above |
| `POLICY` | `policies/$(SYS)/$(SPEC).npz` | proposer to evaluate or render |
| `SEEDS` | `10` | start states per uncertainty level |
| `LEVEL`, `SEED` | `2`, `0` | uncertainty level and start state of `make video` |

### Train a Proposer from Scratch

Training uses JAX on the GPU if one is available (`make build-gpu`, `make run GPU=1
IMAGE_NAME=lirra:cuda`); expert data generation calls Gurobi on the CPU.

```bash
make dataset SYS=x2 SPEC=theta_plus EPISODES=600   # expert synthesis solutions
make bc SYS=x2 SPEC=theta_plus EPOCHS=300          # behavior cloning (blimp: EPOCHS=60)
make pipeline SYS=x2 SPEC=theta_plus               # DAgger, margin-weighted cloning, PPO + evaluations
```

`make pipeline` runs `scripts/dagger_ppo_pipeline.sh`, which writes the margin-weighted cloning
policies to `runs/patrol_bc/` and the PPO policies to `runs/patrol_ppo/`; the paper uses the
policy tagged `dg_ppoC`.
