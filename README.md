<div align="center">

Megatron-LM & Megatron Core
===========================

<h4>GPU-optimized library for training transformer models at scale</h4>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://docs.nvidia.com/Megatron-Core/developer-guide/latest/index.html)
[![version](https://img.shields.io/badge/release-0.12.0-green)](./CHANGELOG.md)
[![license](https://img.shields.io/badge/license-Apache-blue)](./LICENSE)

<div align="left">

> ## 🚨 **DEVELOPMENT BRANCH**
> ⚠️ **EXPERIMENTAL FEATURES** - This is the **dev branch** with experimental features. 
>
> **→ For releases and comprehensive documentation, visit the [main branch](https://github.com/NVIDIA/Megatron-LM)**

## ⚡ Quickstart

```bash
# Clone the dev branch
git clone -b dev https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout dev
git checkout 7d1c016856bb457ae2ff88ad502b8ea9aa3d31fa
pip install --no-build-isolation .[mlm,dev]
cd ../Emerging-Optimizers
git checkout 4d84ccadc2bb2ef1f9d346db443562bb44255070
cp -r emerging_optimizers ../Megatron-LM
cd -
```

<details>
<summary>Table of Contents</summary>

**Getting Started**
- [Megatron-LM \& Megatron Core](#megatron-lm--megatron-core)
  - [⚡ Quickstart](#-quickstart)
  - [Dev Branch Philosophy](#dev-branch-philosophy)
    - [Fast Iteration](#fast-iteration)
    - [Feature Lifecycle (Coming Soon)](#feature-lifecycle-coming-soon)
    - [Stability Expectations](#stability-expectations)
  - [Performance \& Benchmarking](#performance--benchmarking)
  - [Community \& Support](#community--support)
    - [Getting Help](#getting-help)
    - [Contributing](#contributing)
    - [Citation](#citation)

**For Complete Documentation** → [Main Branch](https://github.com/NVIDIA/Megatron-LM) | [Official Docs](https://docs.nvidia.com/Megatron-Core/)

</details>






## Dev Branch Philosophy

### Fast Iteration
- **Streamlined Review**: 1 code owner + 1 dev approver (can delegate review) + CI/CD

### Feature Lifecycle (Coming Soon)
- **6-Month Timeline**: Experimental features must graduate to stable or be deprecated
- **Migration Support**: Assistance provided for feature transitions

### Stability Expectations
- **Experimental Nature**: Features may change or be removed as development progresses
- **Testing**: All features will pass convergence and performance validation before inclusion
- **Support**: Dev branch issues should include `[DEV]` prefix

## Performance & Benchmarking

- 🚀 [2025/11] [Optimizing DeepSeek-V3 Training Performance on NVIDIA GB200 NVL72](docs/discussions/deepseek-v3-gb200-optimization/deepseek-v3-gb200-optimization.md).
- ⚡ [2025/11] [A Guide to Reproduce DeepSeek-V3 Pre-training Performance on GB200](docs/discussions/deepseek-v3-gb200-optimization/deepseek-v3-gb200-reproduce-guide.md).

## Community & Support

### Getting Help
- 📖 **[Documentation](https://docs.nvidia.com/Megatron-Core/)** - Official documentation
- 🐛 **[Issues](https://github.com/NVIDIA/Megatron-LM/issues)** - Bug reports and feature requests

### Contributing
We ❤️ contributions! Ways to contribute:

- 🐛 **Report bugs** - Help us improve reliability
- 💡 **Suggest features** - Shape the future of Megatron Core
- 📝 **Improve docs** - Make Megatron Core more accessible
- 🔧 **Submit PRs** - Contribute code improvements

**→ [Contributing Guide](./CONTRIBUTING.md)**

### Citation
```bibtex
@article{megatron-lm,
  title={Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism},
  author={Shoeybi, Mohammad and Patwary, Mostofa and Puri, Raul and LeGresley, Patrick and Casper, Jared and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:1909.08053},
  year={2019}
}
```
