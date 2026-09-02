# FrameTrain

**Local AI model training & fine-tuning — a cross-platform desktop app.**

[![Website](https://img.shields.io/badge/website-frame--train.com-6d28d9)](https://frame-train.com)
[![Download](https://img.shields.io/badge/download-Windows%20%C2%B7%20macOS%20%C2%B7%20Linux-1d4ed8)](https://frame-train.com/en/download)
[![Version](https://img.shields.io/badge/version-1.2.45-16a34a)](https://frame-train.com/en/changelog)
![Built with Tauri + React](https://img.shields.io/badge/built%20with-Tauri%202%20%2B%20React-333)

FrameTrain is a desktop application for training and fine-tuning AI models **locally on your own hardware** — no cloud required. It brings LoRA/QLoRA fine-tuning of Hugging Face models, PyTorch-based training with GPU acceleration (NVIDIA CUDA **and** Apple Silicon / Metal MPS), a visual neural-network builder, dataset management, live training monitoring and automatic model versioning together in one GUI.

Because training runs entirely on your machine, your datasets and models never leave your device — GDPR-compliant by design.

> This is the official source repository for the FrameTrain desktop app.
> Ready-to-install builds are on the [website](https://frame-train.com/en/download) and under [Releases](https://github.com/FrameSphere/FrameTrain-App/releases).

## Features

**Fine-tuning & training**
- LoRA and QLoRA fine-tuning of large language models
- PyTorch-based training pipeline
- Import Hugging Face models or load local ones
- Export trained models, including GGUF for local inference
- [Ollama](https://ollama.com) integration for running results locally

**Synapse Builder — visual neural networks**
- Build custom architectures on a drag-and-drop canvas (powered by React Flow)
- Transformer, CNN and LSTM building blocks
- Train directly from the canvas

**Data & workflow**
- Dataset import, Parquet support, train/validation/test splitting
- Live training monitoring with loss curves
- AI training analysis & coach for diagnosing runs
- Automatic model versioning and comparison of training runs

**Runs everywhere, stays private**
- Windows, macOS and Linux
- NVIDIA CUDA and Apple Silicon (M1/M2/M3/M4 via Metal MPS) — no CUDA required on Mac
- Fully local and offline

See the app in action on the [screenshots tour](https://frame-train.com/en/screenshots).

## Download

Get the latest build for your platform:

- **[frame-train.com/download](https://frame-train.com/en/download)** — recommended
- **[GitHub Releases](https://github.com/FrameSphere/FrameTrain-App/releases)** — all versions

Requirements: a 64-bit OS; for LLM fine-tuning an NVIDIA GPU or an Apple Silicon Mac is recommended (8 GB unified memory minimum, 16 GB+ for 7B models). See the [Apple Silicon guide](https://frame-train.com/en/apple-silicon) for what your Mac can handle.

## Build from source

FrameTrain is a [Tauri 2](https://tauri.app) app with a React + TypeScript frontend.

**Prerequisites**
- [Node.js](https://nodejs.org) 18+ and npm
- [Rust](https://www.rust-lang.org/tools/install) (stable) — required by Tauri
- Platform tooling for Tauri (see [Tauri prerequisites](https://tauri.app/start/prerequisites/))

**Setup**

```bash
# install frontend dependencies
npm install

# run the desktop app in development (hot reload)
npm run tauri:dev

# build a production desktop binary for your platform
npm run tauri:build
```

Frontend-only commands (without the desktop shell) are also available:

```bash
npm run dev       # Vite dev server
npm run build     # type-check + build the web frontend
npm run preview   # preview the built frontend
```

## Tech stack

- **Shell:** Tauri 2 (Rust) — small, secure, native binaries
- **Frontend:** React 18, TypeScript, Vite, Tailwind CSS
- **Visual builder:** React Flow (`@xyflow/react`)
- **Training core:** PyTorch, Hugging Face, PEFT/LoRA, QLoRA, BitsAndBytes
- **Testing:** Vitest, Testing Library

## Project structure

```
src/          React + TypeScript frontend (UI, AI logic, datasets, Synapse Builder)
src-tauri/    Rust backend (native shell, AI proxy, file system, updater)
scripts/      Build and maintenance scripts
```

## Learn more

- [Documentation](https://frame-train.com/en/docs) · [Guides](https://frame-train.com/en/guides)
- [LoRA fine-tuning guide](https://frame-train.com/en/guides/lora-finetuning)
- [FrameTrain vs. MLX LoRA Studio vs. Unsloth](https://frame-train.com/en/compare)
- [FAQ](https://frame-train.com/en/faq) · [Changelog](https://frame-train.com/en/changelog)

## License

FrameTrain is proprietary software. See [LICENSE](./LICENSE) for the terms that apply to this repository.

---

Made by [FrameSphere](https://frame-train.com).
