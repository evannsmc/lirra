IMAGE_NAME=lirra
DOCKERHUB_IMAGE=evannsmc/lirra:latest
CONTAINER_NAME=lirra
PROJECT_ROOT:=$(abspath ..)
GUROBI_LIC ?= $(HOME)/gurobi.lic
HOST_UID := $(shell id -u)
HOST_GID := $(shell id -g)
DOCKER_USER := $(HOST_UID):$(HOST_GID)
GPU ?= 0
GPU_FLAGS := $(if $(filter 1,$(GPU)),--gpus all,)
LIC_MOUNT := $(if $(wildcard $(GUROBI_LIC)),-v $(GUROBI_LIC):/opt/gurobi/gurobi.lic:ro,)

# Experiment parameters
SYS ?= x2
SPEC ?= theta_plus
LEVEL ?= 2
SEED ?= 0
SEEDS ?= 10
POLICY ?= policies/$(SYS)/$(SPEC).npz
EXEC = docker exec -it --user $(DOCKER_USER) -e HOME=/tmp $(CONTAINER_NAME) bash -lc
RUN_IN = cd /workspace/lirra &&

.PHONY: build build-gpu run start stop kill attach pull push test eval eval-nominal video table \
        dataset bc pipeline paper-evals

# ------------------------------------------------------------- container lifecycle
build:
	docker build -f Dockerfile .. -t $(IMAGE_NAME)

build-gpu:
	docker build -f Dockerfile .. --build-arg JAX_EXTRA=cuda12 -t $(IMAGE_NAME):cuda

run:
	@[ -n "$(LIC_MOUNT)" ] || echo "note: no Gurobi licence at $(GUROBI_LIC); X2 and blimp synthesis needs one (see README)"
	docker run -itd --rm --ipc host --shm-size 8g $(GPU_FLAGS) \
		--user $(DOCKER_USER) -e HOME=/tmp \
		-v $(PROJECT_ROOT):/workspace/lirra $(LIC_MOUNT) \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME)

start:
	docker start $(CONTAINER_NAME)

stop:
	docker stop $(CONTAINER_NAME)

kill:
	docker kill $(CONTAINER_NAME)

attach:
	docker exec -it --user $(DOCKER_USER) -e HOME=/tmp $(CONTAINER_NAME) bash

pull:
	docker pull $(DOCKERHUB_IMAGE)
	docker tag $(DOCKERHUB_IMAGE) $(IMAGE_NAME)

push:
	docker tag $(IMAGE_NAME) $(DOCKERHUB_IMAGE)
	docker push $(DOCKERHUB_IMAGE)

# ------------------------------------------------------------- experiments
test:
	$(EXEC) "$(RUN_IN) python -m pytest -n 4 -m 'not slow' -q"

# Standard I-STL (always-on), LIRRA (checked) and the proposer alone on 10 starts x 3 uncertainty levels.
# Tags follow the paper's evaluations, so `make table` picks the runs up.
eval:
	$(EXEC) "$(RUN_IN) for pair in always_on:fu_always_on shield:fu_ppoC_shield policy_only:fu_ppoC_policy_only; do \
	    python experiments/patrol/evaluate.py --system $(SYS) --spec $(SPEC) --seeds $(SEEDS) --reference clock \
	        --policy $(POLICY) --pairs \$${pair%%:*} --tag $(SPEC)_\$${pair#*:}; done"

# Uncertain-boundary specs (*_u): always-on synthesis with the nominal boundaries, scored against the true region
eval-nominal:
	$(EXEC) "$(RUN_IN) python experiments/patrol/evaluate.py --system $(SYS) --spec $(SPEC) --seeds $(SEEDS) \
	    --reference clock --pairs always_on --nominal-predicates --tag $(SPEC)_fu_always_on_nominal"

video:
	$(EXEC) "$(RUN_IN) python experiments/patrol/render_episode.py --system $(SYS) --spec $(SPEC) \
	    --policy $(POLICY) --level $(LEVEL) --seed $(SEED) --view composite mujoco"

table:
	$(EXEC) "$(RUN_IN) python experiments/patrol/main_table.py --blimp-runs fu"

# All evaluations behind the paper's table (both vehicles, the four cube specifications). Hours on a CPU.
paper-evals:
	for s in x2 blimp; do for p in theta_plus phi_plus theta_plus_u phi_plus_u; do \
	    $(MAKE) eval SYS=$$s SPEC=$$p; done; \
	    for p in theta_plus_u phi_plus_u; do $(MAKE) eval-nominal SYS=$$s SPEC=$$p; done; done
	$(MAKE) table

# ------------------------------------------------------------- training from scratch (GPU recommended)
EPISODES ?= 600
dataset:
	$(EXEC) "$(RUN_IN) python experiments/patrol/dataset.py --system $(SYS) --spec $(SPEC) --reference clock \
	    --episodes $(EPISODES) --tag $(SPEC)_clock"

EPOCHS ?= 300
bc:
	$(EXEC) "$(RUN_IN) ds=\$$(ls -d runs/patrol_dataset/$(SYS)/*_$(SPEC)_clock | tail -1); \
	    python experiments/patrol/train_bc.py --system $(SYS) --spec $(SPEC) --reference clock \
	        --data \$$ds/dataset.npz --epochs $(EPOCHS) --tag $(SPEC)_clock_bc"

# DAgger rounds, margin-weighted cloning, PPO; uses the newest dataset and BC policy of SYS/SPEC
pipeline:
	$(EXEC) "$(RUN_IN) ds=\$$(ls -d runs/patrol_dataset/$(SYS)/*_$(SPEC)_clock | tail -1); \
	    bc=\$$(ls -d runs/patrol_bc/$(SYS)/*_$(SPEC)_clock_bc | tail -1); \
	    scripts/dagger_ppo_pipeline.sh $(SYS) $(SPEC) \$$bc/policy.npz \$$ds/dataset.npz"
