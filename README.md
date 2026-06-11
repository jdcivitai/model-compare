# Model Compare Extension

Generate images with the same prompt across multiple checkpoints and display the results side by side for easy comparison.

## Features

- **Multi-model comparison** — Select any number of checkpoints and generate images for all of them in one click
- **Per-model presets** — Each checkpoint remembers its own optimal settings (saved in `presets/`)
- **Multi-LoRA support** — Apply multiple LoRAs at once with a shared strength value
- **LoRA search** — Filter the LoRA list by typing to quickly find what you need
- **Unload All Models** — Free up VRAM by unloading every loaded model and clearing caches
- **Zero core modifications** — Works as a drop-in extension, no changes to core files needed

## Installation

### Clone from GitHub

```bash
cd extensions
git clone https://github.com/jdcivitai/model-compare.git
```

### Full Setup (with preset save/load in main UI)

The extension works standalone, but for the full experience — saving presets from the main txt2img tab and auto-loading them when you switch checkpoints — you also need the preset system installed in the main UI:

1. **Copy `modules/presets.py`** into your WebUI's `modules/` directory
2. **Modify `modules/ui.py`** to add the Save Preset button and checkpoint-change wiring (see [full diff](https://github.com/jdcivitai/model-compare#full-setup) for details)
3. **Add CSS** for the Save Preset button to `style.css`:

```css
#save_preset_btn {
    min-width: 120px !important;
}
```

Without the main UI preset system, the Model Compare tab still works — it just won't auto-load presets when switching checkpoints in the main UI, and there's no Save Preset button. Presets are still read from `presets/` during comparison generations.

## Usage

1. Open the **Model Compare** tab
2. Check the models you want to compare
3. Enter a prompt (or leave blank to use each model's preset)
4. Optionally select LoRAs and adjust strength
5. Click **Generate Comparison**

### Presets

Each model's optimal settings are saved as JSON files in the `presets/` directory. In the Model Compare tab, if you leave the prompt blank, each model will use its saved preset prompt and parameters. The sampling settings shown in the collapsible section are **fallback defaults** — they're only used if the model has no preset.

## Requirements

- Stable Diffusion WebUI reForge (https://github.com/Panchovix/stable-diffusion-webui-reForge)
- No additional Python packages required

## How It Works

The extension uses a `NullScripts` wrapper object to safely bypass the script system during comparison generations, ensuring complete isolation between models without modifying any core files. Each model is loaded, generated, and unloaded sequentially with full cache clearing between generations to prevent state leakage.
