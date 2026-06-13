"""
Model Compare Extension for Stable Diffusion WebUI reForge
Generates images with the same prompt across multiple checkpoints
and displays results side by side for comparison.

Install: Clone/copy this extension into extensions/model-compare/
No core file modifications required.
"""

import os
import re
import sys
import gc
import torch
import gradio as gr

from modules import shared, sd_models, sd_samplers, sd_schedulers, processing, call_queue, script_callbacks
from modules_forge import main_thread
from ldm_patched.modules import model_management


# ─── Preset Management ────────────────────────────────────────────────────────
# Try to use the main UI's preset system (modules/presets.py) if available.
# Otherwise fall back to the extension's own preset code using the same directory.


def _try_import_main_ui_presets():
    """Attempt to import load_preset from the main UI's preset module."""
    try:
        from modules.presets import load_preset as lp
        return lp
    except ImportError:
        return None


_main_ui_load_preset = _try_import_main_ui_presets()

if _main_ui_load_preset is not None:
    # Use the main UI's preset system directly
    load_preset = _main_ui_load_preset
else:
    # Fall back to internal preset system — same directory as the main UI uses
    _presets_dir = os.path.join(shared.script_path, "presets")

    DEFAULT_PRESET = {
        "prompt": "",
        "negative_prompt": "",
        "sampler_name": "Euler a",
        "scheduler": "Automatic",
        "steps": 20,
        "cfg_scale": 7.0,
        "width": 512,
        "height": 512,
        "batch_count": 1,
        "clip_skip": 1,
    }

    def _ensure_presets_dir():
        os.makedirs(_presets_dir, exist_ok=True)

    def _normalize_checkpoint_name(checkpoint_name):
        base = checkpoint_name
        bracket_idx = checkpoint_name.rfind(" [")
        if bracket_idx > 0:
            base = checkpoint_name[:bracket_idx]
        safe_name = base
        for ch in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
            safe_name = safe_name.replace(ch, "_")
        return safe_name

    def _preset_filename(checkpoint_name):
        safe_name = _normalize_checkpoint_name(checkpoint_name)
        return os.path.join(_presets_dir, f"{safe_name}.json")

    def load_preset(checkpoint_name):
        """Load a preset for the given checkpoint. Returns dict or defaults."""
        _ensure_presets_dir()
        filepath = _preset_filename(checkpoint_name)
        if os.path.exists(filepath):
            try:
                import json
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                result = DEFAULT_PRESET.copy()
                result.update(data)
                return result
            except Exception as e:
                print(f"[Model Compare] Warning: Failed to load preset for '{checkpoint_name}': {e}")
        return DEFAULT_PRESET.copy()



# ─── NullScripts Wrapper ──────────────────────────────────────────────────────
# Replaces p.scripts = None with a no-op object so processing.py doesn't crash.
# This avoids the need to modify core files.


class NullScripts:
    """No-op scripts object that safely bypasses the script system."""
    alwayson_scripts = []

    def setup_scrips(self, p, **kwargs):
        pass

    def before_process(self, p, **kwargs):
        pass

    def process(self, p, **kwargs):
        pass

    def before_process_batch(self, p, **kwargs):
        pass

    def process_before_every_sampling(self, p, **kwargs):
        pass

    def process_batch(self, p, **kwargs):
        pass

    def post_sample(self, p, **kwargs):
        pass

    def postprocess_batch(self, p, **kwargs):
        pass

    def postprocess_batch_list(self, p, **kwargs):
        pass

    def postprocess_image(self, p, **kwargs):
        pass

    def postprocess_maskoverlay(self, p, **kwargs):
        pass

    def postprocess_image_after_composite(self, p, **kwargs):
        pass

    def postprocess(self, p, **kwargs):
        pass

    def after_extra_networks_activate(self, p, **kwargs):
        pass

    def before_hr(self, p, **kwargs):
        pass

    def before_process_init_images(self, p, **kwargs):
        pass

    def on_mask_blend(self, p, **kwargs):
        pass

    def process_before_every_step(self, p, **kwargs):
        pass

    def setup(self, p, **kwargs):
        pass


# ─── Helpers ──────────────────────────────────────────────────────────────────


refresh_symbol = "\U0001f504"  # 🔄


def _checkpoint_display_names():
    """Return list of checkpoint display names (without hash suffixes)."""
    names = []
    for info in sd_models.checkpoints_list.values():
        names.append(info.name_for_extra)
    return sorted(set(names))


def _lora_names():
    """Return list of LoRA/LyCORIS names using recursive scanning."""
    allowed_extensions = {".pt", ".ckpt", ".safetensors"}
    names = set()

    lora_dir = shared.cmd_opts.lora_dir
    for full_path in shared.walk_files(lora_dir, allowed_extensions=allowed_extensions):
        name = os.path.splitext(os.path.basename(full_path))[0]
        names.add(name)

    lyco_dir = getattr(shared.cmd_opts, 'lyco_dir', None)
    if lyco_dir:
        for full_path in shared.walk_files(lyco_dir, allowed_extensions=allowed_extensions):
            name = os.path.splitext(os.path.basename(full_path))[0]
            names.add(name)

    return sorted(names)


def _inject_loras_into_prompt(prompt, selected_loras, lora_strength):
    """Inject LoRA tags into prompt for each selected LoRA."""
    if not selected_loras:
        return prompt
    prompt = re.sub(r"<lora:[^>]+>", "", prompt).strip()
    tags = " ".join(f"<lora:{name}:{lora_strength}>" for name in selected_loras)
    return f"{tags} {prompt}"


def _plaintext_to_html(text, classname=None):
    """Convert plain text to HTML paragraph."""
    import html as html_module
    content = "<br>\n".join(html_module.escape(x) for x in text.split('\n'))
    if classname:
        return f"<p class='{classname}'>{content}</p>"
    return f"<p>{content}</p>"


def _unload_all_models():
    """Unload all loaded models and clear caches."""
    print("[Model Compare] Unloading all models...")

    while sd_models.model_data.loaded_sd_models:
        model = sd_models.model_data.loaded_sd_models.pop()
        if hasattr(model, 'model_unload'):
            model.model_unload()
        elif hasattr(model, 'to') and hasattr(model, 'offload_device'):
            model.to(model.offload_device)
        elif hasattr(model, 'to'):
            model.to('cpu')

    sd_models.model_data.sd_model = None

    model_management.soft_empty_cache(force=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

    processing.StableDiffusionProcessing.cached_c = [None, None]
    processing.StableDiffusionProcessing.cached_uc = [None, None]

    print("[Model Compare] All models unloaded and caches cleared")


def _clear_generation_caches():
    """Clear all caches between model generations to prevent state leakage."""
    processing.StableDiffusionProcessing.cached_c = [None, None]
    processing.StableDiffusionProcessing.cached_uc = [None, None]

    processing.StableDiffusionProcessingTxt2Img.cached_c = [None, None]
    processing.StableDiffusionProcessingTxt2Img.cached_uc = [None, None]
    processing.StableDiffusionProcessingTxt2Img.cached_hr_c = [None, None]
    processing.StableDiffusionProcessingTxt2Img.cached_hr_uc = [None, None]

    model_management.soft_empty_cache(force=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


# ─── Generation ───────────────────────────────────────────────────────────────


def _generate_single(checkpoint_info, prompt, negative_prompt,
                     sampler_name, scheduler, steps, cfg_scale,
                     width, height, batch_count, clip_skip,
                     selected_loras, lora_strength):
    """Generate a single image with a specific checkpoint."""

    _clear_generation_caches()

    try:
        sd_models.reload_model_weights(info=checkpoint_info)
    except Exception as e:
        return [], f"Error loading checkpoint: {e}", checkpoint_info.title

    preset = load_preset(checkpoint_info.title)

    final_prompt = prompt if prompt.strip() else preset.get("prompt", "")
    final_negative_prompt = negative_prompt if negative_prompt.strip() else preset.get("negative_prompt", "")

    final_sampler = preset.get("sampler_name", sampler_name)
    final_scheduler = preset.get("scheduler", scheduler)
    final_steps = int(preset.get("steps", steps))
    final_cfg = float(preset.get("cfg_scale", cfg_scale))
    final_width = int(preset.get("width", width))
    final_height = int(preset.get("height", height))
    final_batch = int(preset.get("batch_count", batch_count))
    final_clip_skip = int(preset.get("clip_skip", 1))

    print(f"[Model Compare] {checkpoint_info.title} -> sampler={final_sampler} steps={final_steps} "
          f"cfg={final_cfg} size={final_width}x{final_height} clip_skip={final_clip_skip}")

    if selected_loras:
        final_prompt = _inject_loras_into_prompt(final_prompt, selected_loras, float(lora_strength))

    original_sd_model_checkpoint = shared.opts.sd_model_checkpoint
    shared.opts.sd_model_checkpoint = checkpoint_info.title

    try:
        override_settings = {}
        if final_clip_skip != 1:
            override_settings["CLIP_stop_at_last_layers"] = final_clip_skip

        p = processing.StableDiffusionProcessingTxt2Img(
            outpath_samples=shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples,
            outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids,
            prompt=final_prompt,
            styles=[],
            negative_prompt=final_negative_prompt,
            batch_size=1,
            n_iter=final_batch,
            cfg_scale=final_cfg,
            width=final_width,
            height=final_height,
            sampler_name=final_sampler,
            scheduler=final_scheduler,
            steps=final_steps,
            seed=-1,
            override_settings=override_settings,
            override_settings_restore_afterwards=True,
        )
        # Use NullScripts wrapper to bypass script system without modifying core files
        p.scripts = NullScripts()

        from contextlib import closing
        with closing(p):
            processed = processing.process_images(p)

    finally:
        shared.opts.sd_model_checkpoint = original_sd_model_checkpoint

    shared.total_tqdm.clear()

    lora_info = f"| LoRA: {', '.join(selected_loras)} ({lora_strength})" if selected_loras else ""
    info_text = (f"**{checkpoint_info.title}**  "
                 f"| Sampler: {final_sampler} | Steps: {final_steps} | CFG: {final_cfg}  "
                 f"| {final_width}x{final_height}"
                 f"{lora_info}")

    title_short = checkpoint_info.short_title if hasattr(checkpoint_info, 'short_title') else checkpoint_info.title
    gallery_items = []
    for img in processed.images + processed.extra_images:
        gallery_items.append((img, title_short))

    return gallery_items, info_text, checkpoint_info.title


def _compare_generate(prompt, negative_prompt, selected_checkpoints,
                      sampler_name, scheduler, steps, cfg_scale,
                      width, height, batch_count, clip_skip,
                      selected_loras, lora_strength):
    """Generate images for all selected checkpoints and return combined results."""

    if not selected_checkpoints:
        return [], [], f"<p style='color:#c00;'>Please select at least one checkpoint.</p>"

    original_checkpoint = shared.opts.sd_model_checkpoint

    all_images = []
    all_infos = []

    for ckpt_display_name in selected_checkpoints:
        ckpt_info = sd_models.checkpoint_aliases.get(ckpt_display_name)
        if ckpt_info is None:
            all_infos.append(f"<p style='color:#c00;'>Checkpoint &lsquo;{ckpt_display_name}&rsquo; not found</p>")
            continue

        gallery_items, info_text, title = _generate_single(
            ckpt_info, prompt, negative_prompt,
            sampler_name, scheduler, steps, cfg_scale,
            width, height, batch_count, clip_skip,
            selected_loras, lora_strength
        )

        all_images.extend(gallery_items)
        all_infos.append(_plaintext_to_html(info_text))

    if original_checkpoint:
        orig_info = sd_models.checkpoints_list.get(original_checkpoint)
        if orig_info:
            try:
                sd_models.reload_model_weights(info=orig_info)
            except Exception:
                pass

    return all_images, all_infos, ""


def compare_generate(prompt, negative_prompt, selected_checkpoints,
                     sampler_name, scheduler, steps, cfg_scale,
                     width, height, batch_count, clip_skip,
                     selected_loras, lora_strength):
    """Wrapper that runs generation inside Forge's main thread."""
    return main_thread.run_and_wait_result(
        _compare_generate,
        prompt, negative_prompt, selected_checkpoints,
        sampler_name, scheduler, steps, cfg_scale,
        width, height, batch_count, clip_skip,
        selected_loras, lora_strength
    )


# ─── UI Builder ───────────────────────────────────────────────────────────────


def create_ui():
    """Build and return the Model Compare UI as a gr.Blocks."""

    checkpoint_display_names = _checkpoint_display_names()
    lora_names = _lora_names()

    with gr.Blocks(analytics_enabled=False) as compare_interface:

        gr.HTML("""
        <div style="text-align:center; padding:8px 0;">
          <h2 style="margin:0;">Model Compare</h2>
          <p style="font-size:0.85em; color:#888; margin:4px 0 0 0;">
            Generate the same prompt across multiple checkpoints and compare results
          </p>
        </div>
        """)

        with gr.Row():

            # ── Left Column: Controls ──
            with gr.Column(scale=2, min_width=340):

                # ── Unload All Models Button ──
                unload_btn = gr.Button(
                    "Unload All Models",
                    variant="secondary",
                    elem_id="compare_unload_all_btn",
                    size="sm",
                )

                def do_unload_all():
                    _unload_all_models()
                    return "All models unloaded. Caches cleared."

                unload_status = gr.HTML(elem_id="compare_unload_status")

                unload_btn.click(
                    fn=do_unload_all,
                    inputs=[],
                    outputs=[unload_status],
                )

                # ── Checkpoint Selection ──
                with gr.Row():
                    checkpoint_group = gr.CheckboxGroup(
                        choices=checkpoint_display_names,
                        value=[],
                        label="Select Checkpoints to Compare",
                        elem_id="compare_checkpoints",
                    )
                    refresh_ckpt_btn = gr.Button(
                        refresh_symbol,
                        elem_id="compare_refresh_checkpoints",
                        tooltip="Refresh checkpoint list",
                    )

                def refresh_checkpoints():
                    sd_models.list_models()
                    names = _checkpoint_display_names()
                    return gr.update(choices=names, value=[])

                refresh_ckpt_btn.click(
                    fn=refresh_checkpoints,
                    inputs=[],
                    outputs=[checkpoint_group],
                )

                # ── Prompts ──
                prompt_box = gr.Textbox(
                    label="Prompt",
                    lines=3,
                    placeholder="Describe what you want to generate...\n(Uses preset prompt if left blank)",
                    elem_id="compare_prompt",
                )
                negative_prompt_box = gr.Textbox(
                    label="Negative Prompt",
                    lines=2,
                    placeholder="Things to avoid...\n(Uses preset negative prompt if left blank)",
                    elem_id="compare_neg_prompt",
                )

                # ── Generate Button (prominent, near top) ──
                generate_btn = gr.Button(
                    "Generate Comparison",
                    variant="primary",
                    elem_id="compare_generate_btn",
                )

                # ── Sampling Parameters (collapsible) ──
                with gr.Accordion("Sampling Settings (fallback defaults)", open=False):
                    sampler_dropdown = gr.Dropdown(
                        choices=[x.name for x in sd_samplers.visible_samplers()],
                        value="Euler a",
                        label="Sampling Method",
                        interactive=True,
                    )
                    scheduler_dropdown = gr.Dropdown(
                        choices=[x.label for x in sd_schedulers.schedulers],
                        value="Automatic",
                        label="Schedule Type",
                        interactive=True,
                    )
                    steps_slider = gr.Slider(
                        minimum=1, maximum=150, step=1,
                        value=20,
                        label="Sampling Steps",
                    )
                    cfg_slider = gr.Slider(
                        minimum=0.1, maximum=30.0, step=0.1,
                        value=7.0,
                        label="CFG Scale",
                    )
                    clip_skip_slider = gr.Slider(
                        minimum=1, maximum=12, step=1,
                        value=1,
                        label="Clip Skip",
                    )

                # ── Dimensions (collapsible) ──
                with gr.Accordion("Dimensions (fallback defaults)", open=False):
                    with gr.Row():
                        width_slider = gr.Slider(
                            minimum=64, maximum=2048, step=8,
                            value=512,
                            label="Width",
                        )
                        height_slider = gr.Slider(
                            minimum=64, maximum=2048, step=8,
                            value=512,
                            label="Height",
                        )
                    batch_count_slider = gr.Slider(
                        minimum=1, maximum=10, step=1,
                        value=1,
                        label="Batch Count per Model",
                    )

                # ── LoRA Selector (collapsible) ──
                with gr.Accordion("LoRA", open=False):
                    lora_search_box = gr.Textbox(
                        label="Search LoRA",
                        placeholder="Type to filter...",
                        lines=1,
                        elem_id="compare_lora_search",
                    )
                    lora_group = gr.CheckboxGroup(
                        choices=lora_names,
                        value=[],
                        label="Select LoRAs",
                        interactive=True,
                        elem_id="compare_lora",
                    )
                    refresh_lora_btn = gr.Button(
                        refresh_symbol,
                        elem_id="compare_refresh_loras",
                        tooltip="Refresh LoRA list",
                    )
                    lora_strength_slider = gr.Slider(
                        minimum=0.0, maximum=2.0, step=0.05,
                        value=1.0,
                        label="LoRA Strength (applied to all selected)",
                    )

                    lora_state = {"names": lora_names}

                    def filter_loras(search_text, current_value):
                        all_names = lora_state["names"]
                        if not search_text or search_text.strip() == "":
                            return gr.update(choices=all_names, value=current_value)
                        search_lower = search_text.lower()
                        filtered = [n for n in all_names if search_lower in n.lower()]
                        kept_value = [v for v in (current_value or []) if v in filtered]
                        return gr.update(choices=filtered, value=kept_value)

                    def refresh_loras(search_text, current_value):
                        names = _lora_names()
                        lora_state["names"] = names
                        if search_text and search_text.strip():
                            search_lower = search_text.lower()
                            filtered = [n for n in names if search_lower in n.lower()]
                            return gr.update(choices=filtered, value=[])
                        return gr.update(choices=names, value=[])

                    lora_search_box.change(
                        fn=filter_loras,
                        inputs=[lora_search_box, lora_group],
                        outputs=[lora_group],
                    )

                    refresh_lora_btn.click(
                        fn=refresh_loras,
                        inputs=[lora_search_box, lora_group],
                        outputs=[lora_group],
                    )

            # ── Right Column: Results ──
            with gr.Column(scale=3, min_width=512):
                gallery = gr.Gallery(
                    label="Comparison Results",
                    show_label=False,
                    elem_id="compare_gallery",
                    columns=2,
                    preview=True,
                    height="auto",
                )
                info_html = gr.HTML(elem_id="compare_info_html")
                error_html = gr.HTML(elem_id="compare_error_html")

        # ── Wire up Generate ──
        wrapped_generate = call_queue.wrap_gradio_gpu_call(
            compare_generate, extra_outputs=[None, "", ""]
        )

        generate_btn.click(
            fn=wrapped_generate,
            inputs=[
                prompt_box,
                negative_prompt_box,
                checkpoint_group,
                sampler_dropdown,
                scheduler_dropdown,
                steps_slider,
                cfg_slider,
                width_slider,
                height_slider,
                batch_count_slider,
                clip_skip_slider,
                lora_group,
                lora_strength_slider,
            ],
            outputs=[
                gallery,
                info_html,
                error_html,
            ],
        )

    return compare_interface


# ─── Tab Registration ─────────────────────────────────────────────────────────

def on_ui_tabs():
    """Register the Model Compare tab with the WebUI."""
    compare_ui = create_ui()
    return [(compare_ui, "Model Compare", "modelcompare")]


# Register the tab
script_callbacks.on_ui_tabs(on_ui_tabs)
