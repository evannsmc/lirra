# LIRRA: environment for every experiment of the paper (CPU JAX by default).
#
# The image holds only the dependencies. The code is not copied in: `make run` mounts your clone at
# /workspace/lirra and PYTHONPATH points at its src/, so code changes need no rebuild. Rebuild only when
# pyproject.toml or uv.lock change.
#
#   docker build -f docker/Dockerfile -t lirra .                        # from the repository root
#   docker build -f docker/Dockerfile --build-arg JAX_EXTRA=cuda12 -t lirra:cuda .   # NVIDIA GPU
FROM ghcr.io/astral-sh/uv:0.8-python3.12-bookworm-slim

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# MuJoCo offscreen rendering (EGL / OSMesa), video encoding, make
RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1 libgl1 libgles2 libosmesa6 libglfw3 \
        ffmpeg make git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The environment lives outside the code, so mounting a clone over /workspace/lirra keeps it.
ENV UV_PROJECT_ENVIRONMENT=/opt/lirra-venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/lirra-venv/bin:$PATH \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    RL_STL_DEVICE=auto \
    XLA_PYTHON_CLIENT_PREALLOCATE=false
# Gurobi finds a licence mounted at its default path /opt/gurobi/gurobi.lic (docker/makefile `make run`);
# without one it falls back to the size-limited pip licence (enough for the double integrator).

WORKDIR /workspace/lirra

ARG JAX_EXTRA=""
# Dependencies only (the lirra package itself comes from the mounted clone, see PYTHONPATH below)
COPY pyproject.toml uv.lock README.md /opt/lirra-deps/
RUN cd /opt/lirra-deps && uv sync --locked --no-install-project ${JAX_EXTRA:+--extra $JAX_EXTRA} && \
    python -c "import mujoco, jax, gurobipy; print('mujoco', mujoco.__version__, '| jax', jax.__version__)" && \
    python -c "from mujoco_playground._src import mjx_env; mjx_env.ensure_menagerie_exists(); print('menagerie', mjx_env.MENAGERIE_PATH)"
ENV PYTHONPATH=/workspace/lirra/src

RUN mkdir -p /opt/gurobi && chmod 777 /workspace/lirra && \
    echo 'export PATH=/opt/lirra-venv/bin:$PATH' > /etc/profile.d/lirra.sh && \
    echo '[ -d /workspace/lirra/src/rl_stl ] || echo "No LIRRA code at /workspace/lirra: start the container with make run from your clone (it mounts the repository)."' >> /etc/bash.bashrc && \
    echo 'echo "LIRRA environment. Gurobi licence: $( [ -f /opt/gurobi/gurobi.lic ] && echo "found at /opt/gurobi/gurobi.lic" || echo "none mounted (size-limited pip licence; mount a WLS licence for the X2 and blimp, see docker/README.md)")"' >> /etc/bash.bashrc

CMD ["/bin/bash"]
