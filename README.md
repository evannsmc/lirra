<p align="center">
  <img src="assets/lirra_logo.svg" width="96" alt="LIRRA logo">
</p>

# LIRRA: Learning-Augmented Interval-STL for Recursively-feasible Runtime Assurance

[![Docker Ready](https://img.shields.io/badge/Docker-evannsmc%2Flirra-blue)](https://hub.docker.com/r/evannsmc/lirra)
[![Project site](https://img.shields.io/badge/Project-evannsmc.github.io%2Frl--stl-blue)](https://evannsmc.github.io/rl-stl/)
[![evannsmc.com](https://img.shields.io/badge/evannsmc.com-Project%20Portfolio-blue)](https://www.evannsmc.com)

Code for the paper **Skip the Solve: Learned Trajectory Proposals for Recursively Feasible
Interval-STL Runtime Assurance** by Evanns Morales-Cuadrado, Xinpei Ni, Glen Chou and Samuel Coogan
(Georgia Institute of Technology), submitted to L4DC 2027.

The project website, with the theory, all results and videos of every specification, is
[evannsmc.github.io/rl-stl](https://evannsmc.github.io/rl-stl/).

**LIRRA** lets a vehicle skip most of the expensive signal temporal logic (STL) synthesis it would
otherwise solve online, without giving up its safety guarantee. At every planning step a learned
policy proposes a whole trajectory in about a millisecond. The proposal is accepted only if it
satisfies every constraint of the interval-STL synthesis problem, including control-invariant
terminal sets. If it is rejected, the vehicle keeps flying its last verified plan while synthesis
runs from the state the vehicle will reach when the solver finishes. Satisfaction and recursive
feasibility therefore hold for any proposer and any solver completion time, while the solver is
needed on only a few percent of planning steps.

<p align="center">
  <img src="assets/x2_theta_plus_composite.png" width="92%" alt="LIRRA on the Skydio X2 around the restricted cube"><br>
  <em>Skydio X2 (MuJoCo) patrolling around the restricted cube under LIRRA. Left: top-down view and the
  plan in force (blue: accepted proposal, yellow: solver, grey: cached). Right: MuJoCo view, the
  realized robustness, and the robustness of each proposal. Videos of all eight specifications are on
  the <a href="https://evannsmc.github.io/rl-stl/case_studies.html">project site</a>.</em>
</p>

<p align="center">
  <img src="assets/lirra_loop.svg" width="70%" alt="LIRRA runtime loop: propose, verify, fall back">
</p>

## Results at a Glance

Always-on interval-STL runtime assurance (synthesis at every step) against LIRRA, 30 episodes per
row, on the paper's four specifications around a 3-D restricted cube, which may be entered only briefly (θ: leave within 4 s; φ: leave
within 3 s and stay out 1 s; ⁺: uncertain cube boundaries).

| System | Spec. | LIRRA fallbacks / step | Time I-STL [ms/step] | Time LIRRA [ms/step] | Violations |
|---|---|---|---|---|---|
| Skydio X2 | θ | .006 | 140.9 | 1.7 | 0/30 |
| | φ | .007 | 151.5 | 2.0 | 0/30 |
| | θ⁺ | .013 | 158.8 | 3.1 | 0/30 |
| | φ⁺ | .011 | 151.2 | 3.0 | 0/30 |
| Blimp (GT-MAB) | θ | .062 | 10.9 | 1.7 | 0/30 |
| | φ | .045 | 12.1 | 1.8 | 0/29 |
| | θ⁺ | .077 | 11.7 | 2.2 | 0/30 |
| | φ⁺ | .041 | 14.0 | 1.9 | 0/23 |

The [case studies](https://evannsmc.github.io/rl-stl/case_studies.html) page has all eight
specifications (square and cube, exact and uncertain boundaries), the ablations and the videos.

## Quick Start with Docker

All experiments run in a Docker image with every dependency installed: Python 3.12, MuJoCo/MJX,
MuJoCo Playground, Brax, JAX, `stlpy` and Gurobi.

### Option A: Pre-built Image (Fastest)

```bash
git clone https://github.com/evannsmc/lirra.git && cd lirra/docker
make pull             # pulls evannsmc/lirra:latest from Docker Hub
make run              # starts the container with the repository mounted
```

### Option B: Build from Source

```bash
git clone https://github.com/evannsmc/lirra.git && cd lirra/docker
make build
make run
```

### Gurobi Licence (you need your own)

LIRRA solves its interval-STL synthesis problems with [Gurobi](https://www.gurobi.com). Gurobi is
commercial software, so **we cannot ship a licence: every user needs their own.** The `gurobipy`
package in the image and in `uv.lock` comes with a free *size-limited* licence. That is enough for
the double integrator, but the Skydio X2 and blimp problems are larger and stop with
`initial synthesis failed` unless a full licence is found. See
[Getting a Gurobi licence](#getting-a-gurobi-licence) below.

### Reproduce a Result

```bash
make test                                  # unit tests
make eval SYS=x2 SPEC=theta_plus           # I-STL vs LIRRA vs unchecked proposer, 30 episodes
make video SYS=x2 SPEC=theta_plus          # render one episode (top-down + MuJoCo composite)
make table                                 # rebuild the results tables from your evaluations
```

See [`docker/README.md`](docker/README.md) for every target, including the full training pipeline
(expert data, behavior cloning, DAgger, margin-weighted cloning, PPO) and the paper's evaluations.

## Without Docker

```bash
./setup.sh                   # uv environment; CUDA 12 JAX if a GPU is visible, otherwise CPU
source .venv/bin/activate
uv run pytest -n 8 -m "not slow"
```

The [installation page](https://evannsmc.github.io/rl-stl/installation.html) lists every pipeline
stage and option.

## Getting a Gurobi Licence

Which licence to get depends on how you run LIRRA:

| You are | Running in Docker | Running without Docker |
|---|---|---|
| at a university (student, faculty, staff) | free **Academic WLS** licence | free **Academic WLS** or **Academic Named-User** licence |
| not at a university | a **commercial** or **evaluation** licence from Gurobi (WLS type for Docker) | same |

Machine-locked licences (such as the Academic Named-User licence) are tied to one computer's host ID
and **do not work inside a container**. WLS (Web License Service) licences are checked online, so
they work anywhere, including Docker.

### Academic WLS licence (recommended, works in Docker)

1. Create a free account at the [Gurobi User Portal](https://portal.gurobi.com/iam/register/) with
   your **university email address** and choose the academic user type.
2. In the portal, open the license request page and request a **WLS Academic** licence (see Gurobi's
   [academic program page](https://www.gurobi.com/academia/academic-program-and-licenses/) for the
   current steps and eligibility).
3. Open the new licence in the portal's licences list and **download** its `gurobi.lic` file. It
   contains three lines, `WLSACCESSID`, `WLSSECRET` and `LICENSEID`.
4. Save the file as `~/gurobi.lic` on the machine that runs Docker.
5. Start the container with `make run`; it mounts `~/gurobi.lic` read-only at Gurobi's default path
   (`/opt/gurobi/gurobi.lic`). To use another location: `make run GUROBI_LIC=/path/to/gurobi.lic`.

Academic WLS licences are time-limited; renew them in the portal when they expire, and download the
file again.

### Academic named-user licence (without Docker only)

Request an **Academic Named-User** licence in the same portal and activate it with the `grbgetkey`
command shown there, **while connected to your university's network**. It writes `gurobi.lic` to your
home directory, where `gurobipy` finds it automatically.

### Check that the licence is found

Inside the container (`make attach`) or in the `uv` environment:

```bash
python -c "import gurobipy as gp; m = gp.Model(); print('Gurobi', gp.gurobi.version(), 'licence OK')"
```

With a full licence, Gurobi's banner names your licence type and holder; with only the
size-limited one it says `Restricted license - for non-production use only`.
`make run` also prints a note if it finds no `~/gurobi.lic`, and the container's shell reports which
licence it found when you attach.

**Never commit a licence file.** It is personal, and `.gitignore` excludes `*.lic`.

## Repository Layout

| Path | Contents |
|---|---|
| `src/rl_stl/` | I-STL synthesis (Gurobi MILP), verification, terminal sets, error-age tubes, runtime loop, proposer, vehicle models |
| `experiments/patrol/` | datasets, behavior cloning, PPO, evaluation, rendering and result tables |
| `scripts/` | the DAgger, margin-weighted cloning and PPO pipeline |
| `policies/` | trained proposers behind the paper's results, one per system and specification |
| `tests/` | unit tests |
| `docker/` | Dockerfile and `make` targets |

## Citation

```bibtex
@inproceedings{moralescuadrado2027skip,
  title     = {Skip the Solve: Learned Trajectory Proposals for Recursively Feasible
               Interval-{STL} Runtime Assurance},
  author    = {Morales-Cuadrado, Evanns and Ni, Xinpei and Chou, Glen and Coogan, Samuel},
  booktitle = {Learning for Dynamics and Control Conference (L4DC)},
  year      = {2027},
  note      = {Under review}
}
```

## License

MIT
