# Deployment proof catalog

| deployment | status | hardware | context proof | record |
| --- | --- | --- | --- | --- |
| [Dedicated DeepSeek + Gemma Swarm](deepseek-gemma4-dedicated/README.md) | selected deployment; exact dedicated engine YAMLs measured; router CPU-mock checked | one four-H200 node per model; NVLink measurements | near-window probes at DeepSeek 1M / Gemma 32K; dedicated C4 throughput controls | [configs, evidence and limits](deepseek-gemma4-dedicated/README.md) |
| [DeepSeek V4.1 + Gemma 4 Swarm TP4](deepseek-gemma4-tp4/README.md) | engine YAMLs measured; Swarm packaging statically checked | four shared H200s with NVLink | four ~1.03M DeepSeek + four ~31K Gemma contexts generating together | [configs and source review](deepseek-gemma4-tp4/README.md) |
| [`glm53-gemma4-6xh200.yaml`](deployments/glm53-gemma4-6xh200.yaml) | historical disjoint control on `hopper-fabric-large`; four-GPU packing pending | larger fabric-connected Hopper venue, six assigned | GLM 129,500-token prompt; Gemma 31,493-token prompt in a 32,768 window; simultaneous | [2026-09-02 record](results/20260902-glm53-gemma4-6xh200/RECORD.md) |
