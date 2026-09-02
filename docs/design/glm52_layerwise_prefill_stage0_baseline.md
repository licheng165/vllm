# GLM-5.2 Layerwise Prefill Stage 0 Baseline

This file freezes the integration inputs before the staged implementation. The
four repositories are independent Git repositories, so later cross-repository
stages are identified by a four-SHA deployment manifest rather than by a
single Git commit.

## Source Baseline

| Repository | Branch | Initial SHA | Reference SHA |
|---|---|---|---|
| vLLM | `glm52-model-port` | `ed07cb71a5c09d37278b3147eb23dd09e433cae5` | `404554c294ab76b9780ed1d1b8440a2782629d25` |
| vLLM-Ascend | `glm52-model-port` | `018bbbbae0e52f385238b517e5ee83f725e3759f` | `928730c79735f321cbe824ae77d28c6cc73d20e6` |
| LMCache | `glm52-model-port` | `ca6f861f2a060620e5f87dd5eb87cd6864462ce4` | `554d2f427b8ec86e23e9c6d8ed16631e9eb1b899` |
| LMCache-Ascend | `glm52-model-port` | `7a4e504cac46458f057c0424e1c8208fb8b52905` | `1c82c04d260f0675be4c1c5d7a2d578c3d6053c5` |

All four worktrees were clean when this baseline was captured.

## Deployment Configuration

The source YAML files are outside a Git repository. Their pre-implementation
SHA-256 values are:

| Node | SHA-256 |
|---|---|
| `7.150.4.174` | `7d7368e768b46a704ccf4d5b60f3ec6206d61aac6b43f2cd65454186d0bafb51` |
| `7.150.5.55` | `8fbba4bc336bc9ff389064c31f3ca82a451fe11962e30391f690799c3b1bceac` |
| `7.150.5.81` | `87a9c76bd7482ba4de14f1d0c960182d5e354609de6bfae1c1b5261286b91e9d` |
| `7.150.1.46` | `0b26fc3ea5ea8cea9f9319eec9b686a296a707ba02ce3f54a902a2a281f84c43` |

## Artifact And KV Evidence

The archived target-machine logs establish the initial runtime topology and
capacity failure:

- `0820-2-log-DSA_cache_registration.txt` records 79 LATENT registrations,
  22 INDEXER registrations, and producer executions
  `0,1,2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74,78`.
- `0831-1-prefiller.txt` records a 103.65 GiB requirement and 20.78 GiB
  available per P worker for a 1,000,000-token configuration.
- `0831-1-decoder.txt` records a 103.65 GiB requirement and 22.89 GiB
  available per D worker for the same configuration.
- The BF16 runtime geometry used by the capacity model is block size 128,
  LATENT width 576, and INDEXER width 128. It yields page sizes 147,456 and
  32,768 bytes and a 294,912-byte shared bundle.

Production code must derive these values from runtime registrations. The
numbers above are deployment goldens, not constants to embed in the allocator.

## Collection Status

The implementation workspace is WSL2 without an NPU, the target model path, or
`torch_npu`. It has Python 3.11.6, CPU-only PyTorch 2.13.0, and Transformers
4.57.6. Consequently, the following Stage 0 evidence must be captured on every
target host before hardware acceptance:

- Model `config.json`, quantization config, and weight-index checksums.
- Editable module paths for `vllm`, `vllm_ascend`, `lmcache`, and
  `lmcache_ascend`.
- PyTorch, torch-npu, CANN, driver, and Transformers versions.
- Actual per-worker KV specs, page sizes, bundle size, and available KV bytes.

Use the following non-mutating collection commands in the deployed container:

```bash
git -C /workspace/fsi_lab/lc/fsi/vllm rev-parse HEAD
git -C /workspace/fsi_lab/lc/fsi/vllm-ascend rev-parse HEAD
git -C /workspace/fsi_lab/lc/fsi/LMCache rev-parse HEAD
git -C /workspace/fsi_lab/lc/fsi/LMCache-Ascend rev-parse HEAD
python -c 'import torch, torch_npu, transformers, vllm, lmcache; print(torch.__version__, torch_npu.__version__, transformers.__version__, vllm.__file__, lmcache.__file__)'
npu-smi info
sha256sum /workspace/models/GLM-5.2-w4a8c8-0723/config.json
sha256sum /workspace/models/GLM-5.2-w4a8c8-0723/*quant*config*.json
sha256sum /workspace/models/GLM-5.2-w4a8c8-0723/*.index.json
sha256sum /workspace/fsi_lab/lc/lmcache_mooncake_config.yaml
```

Any target registration other than 79/22, or page geometry other than
147,456/32,768 bytes, invalidates the current capacity calculations and blocks
hardware rollout until the topology namespace and budgets are regenerated.
