# Model Compare Extension

Generate images with the same prompt across multiple checkpoints and display the results side by side for easy comparison.

## Features

- **Multi-model comparison** — Select any number of checkpoints and generate images for all of them in one click
- **Per-model presets** — Each checkpoint remembers its own optimal settings (saved in `configs/presets/`)
- **Multi-LoRA support** — Apply multiple LoRAs at once with a shared strength value
- **LoRA search** — Filter the LoRA list by typing to quickly find what you need
- **Unload All Models** — Free up VRAM by unloading every loaded model and clearing caches
- **Zero core modifications** — Works as a drop-in extension, no changes to core files needed

## Installation

### Option 1: Clone from GitHub

```bash
cd extensions
git clone https://github.com/your-username/model-compare.git
```

### Option 2: Manual Install

1. Download the extension files
2. Place them in `extensions/model-compare/` with this structure:

```
extensions/model-compare/
├── scripts/
│   └── model_compare.py
├── style.css
└── README.md
```

3. Restart the WebUI

## Usage

1. Open the **Model Compare** tab
2. Check the models you want to compare
3. Enter a prompt (or leave blank to use each model's preset)
4. Optionally select LoRAs and adjust strength
5. Click **Generate Comparison**

### Presets

Each model's optimal settings are saved automatically when you generate from the main txt2img tab. In the Model Compare tab, if you leave the prompt blank, each model will use its saved preset prompt and parameters. The sampling settings shown in the collapsible section are **fallback defaults** — they're only used if the model has no preset.

Presets are stored in `configs/presets/` as JSON files.

## Requirements

- Stable Diffusion WebUI reForge (https://github.com/Panchovix/stable-diffusion-webui-reForge)
- No additional Python packages required

## How It Works

The extension uses a `NullScripts` wrapper object to safely bypass the script system during comparison generations, ensuring complete isolation between models without modifying any core files. Each model is loaded, generated, and unloaded sequentially with full cache clearing between generations to prevent state leakage.
