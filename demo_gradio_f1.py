from diffusers_helper.hf_login import login

import os
import requests

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from transformers import AutoProcessor, AutoModelForCausalLM, pipeline, AutoTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket


parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()

# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags

print(args)

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

# Load Florence-2 model for captioning
florence_processor = None
florence_model = None

# Load TinyLlama model for prompt enhancement
tinyllama_model = None
tinyllama_tokenizer = None

def load_tinyllama():
    global tinyllama_model, tinyllama_tokenizer
    if tinyllama_model is None or tinyllama_tokenizer is None:
        try:
            print("Loading TinyLlama model for prompt enhancement...")
            # TinyLlama is an open model that doesn't require login
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            tinyllama_tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Load without quantization
            tinyllama_model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("TinyLlama model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading TinyLlama model: {e}")
            # If we can't load TinyLlama, fall back to an even simpler rule-based enhancer
            return False
    return True

def load_florence():
    global florence_processor, florence_model
    if florence_processor is None or florence_model is None:
        try:
            print("Loading Florence-2 model for captioning...")
            florence_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
            florence_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch.float16, trust_remote_code=True)
            florence_model.to(cpu)
            print("Florence-2 model loaded successfully")
        except Exception as e:
            print(f"Error loading Florence-2 model: {e}")
            return False
    return True

def format_caption_for_video(caption):
    """
    Prepares the raw Florence-2 caption for video generation by cleaning it up
    and adding a simple transition phrase.
    """
    if not caption or caption == "No caption generated":
        return "A person in motion, first standing poised, then beginning to dance with flowing movements, arms gracefully extending outward, followed by a gentle spin."
    
    # Clean up the caption by removing common introductory phrases
    intro_phrases = [
        "The image shows ", "The image is a portrait of ", "The image depicts ", 
        "This image shows ", "The photo shows ", "The picture shows ",
        "This picture shows ", "The image contains ", "The photo is of ", 
        "The image features "
    ]
    
    cleaned_caption = caption
    for phrase in intro_phrases:
        if cleaned_caption.startswith(phrase):
            cleaned_caption = cleaned_caption[len(phrase):]
            break
    
    # Capitalize the first letter if needed
    if cleaned_caption and cleaned_caption[0].islower():
        cleaned_caption = cleaned_caption[0].upper() + cleaned_caption[1:]
    
    # Just add a simple transition phrase to suggest motion
    prompt = f"{cleaned_caption} The scene comes to life with natural movements and animations, Fluid motion and lifelike expressions."
    
    return prompt

@torch.no_grad()
def generate_caption(image):
    if not load_florence():
        return "Failed to load Florence-2 model for captioning"
    
    try:
        florence_model.to(gpu)
        
        # First get a detailed caption of the image
        caption_prompt = "<MORE_DETAILED_CAPTION>"
        inputs = florence_processor(text=caption_prompt, images=image, return_tensors="pt").to(gpu, torch.float16)
        
        generated_ids = florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False
        )
        
        generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_caption = florence_processor.post_process_generation(generated_text, task="<MORE_DETAILED_CAPTION>")
        raw_caption = parsed_caption.get("<MORE_DETAILED_CAPTION>", "No caption generated")
        
        # Now ask Florence to create a video prompt based on the caption
        video_prompt_instruction = f"Based on this image description: '{raw_caption}', create a detailed 2-3 sentence prompt describing how this scene would naturally animate in a short video clip. Include specific motions, expressions, and transitions that would make sense for the subject."
        
        inputs = florence_processor(text=video_prompt_instruction, images=image, return_tensors="pt").to(gpu, torch.float16)
        
        generated_ids = florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=True,
            temperature=0.7
        )
        
        generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # Try to extract the prompt from Florence's response
        try:
            # This is a simple extraction - Florence might format its response in different ways
            lines = [line for line in generated_text.split('\n') if line.strip()]
            video_prompt = ""
            for line in lines:
                if ":" not in line[:20] and "<" not in line and ">" not in line:
                    video_prompt += line + " "
            
            # If we couldn't extract a good prompt, fall back to formatting the original caption
            if len(video_prompt.strip()) < 20:
                video_prompt = format_caption_for_video(raw_caption)
        except:
            # If anything goes wrong in extraction, fall back to the simple format
            video_prompt = format_caption_for_video(raw_caption)
        
        florence_model.to(cpu)
        torch.cuda.empty_cache()
        
        return video_prompt
    except Exception as e:
        print(f"Error generating caption: {e}")
        if florence_model is not None:
            florence_model.to(cpu)
        torch.cuda.empty_cache()
        return "Error generating caption"

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePack_F1_I2V_HY_20250503', torch_dtype=torch.bfloat16).cpu()

vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

stream = AsyncStream()

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)


@torch.no_grad()
def enhance_prompt(prompt_text):
    """
    Enhance a prompt for video generation using TinyLlama model.
    """
    if not prompt_text or prompt_text.strip() == "":
        return "A beautiful, detailed scene with natural elements and interesting lighting."
    
    if not load_tinyllama():
        # Fall back to a simple enhancement if model loading fails
        return f"{prompt_text.strip()}, with smooth motion, dynamic lighting, and cinematic quality."
    
    try:
        # Create the instruction for TinyLlama - specifically asking for shorter prompts with no brackets
        messages = [
            {"role": "system", "content": "You are a video prompt engineer. Your job is to enhance text prompts to create better AI-generated videos."},
            {"role": "user", "content": f"""Enhance this prompt to create a more detailed and dynamic video scene:
"{prompt_text.strip()}"

Create a CONCISE enhanced prompt (3-4 sentences maximum) with:
- Movement and motion details
- Lighting and atmosphere
- Visual quality descriptions

IMPORTANT:
- Keep it SHORT (max 100 words)
- Do NOT use brackets or parentheses
- Do NOT include notes or explanations
- Just write the enhanced prompt directly

Respond ONLY with the enhanced prompt text."""}
        ]
        
        # Convert messages to chat format for TinyLlama
        prompt = tinyllama_tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize and generate
        inputs = tinyllama_tokenizer(prompt, return_tensors="pt").to(tinyllama_model.device)
        
        # Generate with a reasonable maximum length but limit to shorter responses
        outputs = tinyllama_model.generate(
            inputs.input_ids,
            max_new_tokens=200,  # Shorter limit
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tinyllama_tokenizer.eos_token_id
        )
        
        # Decode the response
        generated_text = tinyllama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "<|assistant|>" in generated_text:
            enhanced_prompt = generated_text.split("<|assistant|>")[1].strip()
        else:
            # Try to extract text after the user message
            user_message = f'"{prompt_text.strip()}"'
            if user_message in generated_text:
                enhanced_prompt = generated_text.split(user_message, 1)[1].strip()
            else:
                # Extract the last part of the text if we can't find clear markers
                enhanced_prompt = generated_text.split("\n")[-1].strip()
        
        # Clean up any quotes that might have been added
        enhanced_prompt = enhanced_prompt.strip('"\'')
        
        # Remove any instruction text that might have been included at the beginning
        instruction_phrases = [
            "to create a more detailed and dynamic video scene:",
            "Enhance this prompt",
            "Here's the enhanced prompt",
            "Enhanced prompt:",
            "Here is the enhanced prompt",
            "The enhanced prompt is",
            "Here is my enhanced version"
        ]
        
        for phrase in instruction_phrases:
            if phrase.lower() in enhanced_prompt.lower():
                # Find the phrase and take everything after it
                pattern_index = enhanced_prompt.lower().find(phrase.lower())
                end_of_phrase = pattern_index + len(phrase)
                # Find the first non-space, non-punctuation character after the phrase
                while end_of_phrase < len(enhanced_prompt) and (enhanced_prompt[end_of_phrase].isspace() or enhanced_prompt[end_of_phrase] in ':,"-'):
                    end_of_phrase += 1
                enhanced_prompt = enhanced_prompt[end_of_phrase:].strip()
        
        # Remove any notes or explanations at the end
        note_phrases = [
            "Note:",
            "Note that",
            "Please note",
            "This prompt",
            "The original prompt",
            "I've added",
            "I've enhanced",
            "I've included"
        ]
        
        for phrase in note_phrases:
            if phrase.lower() in enhanced_prompt.lower():
                # Find the phrase and take everything before it
                pattern_index = enhanced_prompt.lower().find(phrase.lower())
                enhanced_prompt = enhanced_prompt[:pattern_index].strip()
        
        # Remove any brackets or parentheses and their contents
        enhanced_prompt = enhanced_prompt.replace('[', '').replace(']', '')
        
        # If the enhanced prompt is too short or seems invalid, fall back to the original
        if len(enhanced_prompt) < len(prompt_text) / 3 or len(enhanced_prompt) < 10:
            # Create a shortened version with basic enhancements
            words = prompt_text.strip().split()
            if len(words) > 50:
                # Take first 50 words if original is too long
                shortened = ' '.join(words[:50])
                enhanced_prompt = f"{shortened}, with smooth motion, dynamic lighting, and cinematic quality."
            else:
                enhanced_prompt = f"{prompt_text.strip()}, with smooth motion, dynamic lighting, and cinematic quality."
        
        return enhanced_prompt
    
    except Exception as e:
        print(f"Error enhancing prompt with TinyLlama: {e}")
        # Fall back to a simple enhancement if processing fails
        words = prompt_text.strip().split()
        if len(words) > 50:
            # Take first 50 words if original is too long
            shortened = ' '.join(words[:50])
            return f"{shortened}, with smooth motion, dynamic lighting, and cinematic quality."
        else:
            return f"{prompt_text.strip()}, with smooth motion, dynamic lighting, and cinematic quality."

@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, resolution, text_to_video_mode=False):
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Processing input or creating random latent for text-to-video
        if text_to_video_mode:
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Creating initial latent ...'))))
            
            # Use resolution from input
            width = height = int(resolution)
            
            # Create a random noise latent as starting point
            # The shape needs to be (1, 16, 1, height//8, width//8) to match history_latents dimension
            rnd_gen = torch.Generator("cuda").manual_seed(seed)
            start_latent = torch.randn(1, 16, 1, height // 8, width // 8, generator=rnd_gen, device=gpu)
            
            # Create dummy image embeddings - the model requires these even for text-to-video
            # The typical shape for SIGLIP embeddings is [1, 257, 1024]
            if not high_vram:
                load_model_as_complete(image_encoder, target_device=gpu)
                
            # Create a blank white image and get its embeddings
            blank_image = np.ones((height, width, 3), dtype=np.uint8) * 255
            image_encoder_output = hf_clip_vision_encode(blank_image, feature_extractor, image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        else:
            # Original image-to-video code path
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

            H, W, C = input_image.shape
            height, width = find_nearest_bucket(H, W, resolution=int(resolution))
            input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

            Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))

            input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
            input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

            # VAE encoding
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

            if not high_vram:
                load_model_as_complete(vae, target_device=gpu)

            start_latent = vae_encode(input_image_pt, vae)

            # CLIP Vision
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

            if not high_vram:
                load_model_as_complete(image_encoder, target_device=gpu)

            image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        if image_encoder_last_hidden_state is not None:
            image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)

        history_latents = torch.zeros(size=(1, 16, 16 + 2 + 1, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None

        history_latents = torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
        total_generated_latent_frames = 1

        for section_index in range(total_latent_sections):
            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

            print(f'section_index = {section_index}, total_latent_sections = {total_latent_sections}')

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds (FPS-30). The video is being extended now ...'
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

            indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
            clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, latent_window_size], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

            clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
            clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=latent_window_size * 4 - 3,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                # shift=3.0,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = latent_window_size * 2
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, -section_latent_frames:], vae).cpu()
                history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')

            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

            stream.output_queue.push(('file', output_filename))
    except:
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    stream.output_queue.push(('end', None))
    return


def process_image_to_video(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, resolution):
    global stream
    assert input_image is not None, 'No input image!'

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

    stream = AsyncStream()

    async_run(worker, input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, resolution, text_to_video_mode=False)

    output_filename = None

    while True:
        flag, data = stream.output_queue.next()

        if flag == 'file':
            output_filename = data
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'progress':
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'end':
            yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
            break


def process_text_to_video(prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, resolution):
    global stream
    assert prompt is not None and prompt.strip() != "", 'No prompt provided!'

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

    stream = AsyncStream()

    # Pass None as input_image for text-to-video mode
    async_run(worker, None, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, resolution, text_to_video_mode=True)

    output_filename = None

    while True:
        flag, data = stream.output_queue.next()

        if flag == 'file':
            output_filename = data
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'progress':
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'end':
            yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
            break


def end_process():
    stream.input_queue.push('end')


def enhance_prompt_event(prompt_text):
    if not prompt_text or prompt_text.strip() == "":
        return "Please enter a prompt first"
    
    enhanced = enhance_prompt(prompt_text)
    return enhanced


@torch.no_grad()
def process_caption(input_image):
    if input_image is None:
        return "Please upload an image first", gr.update(), "", ""
    
    try:
        # Display progress information
        progress_html = make_progress_bar_html(0, 'Loading Florence-2 model...')
        
        if not load_florence():
            return "Failed to load Florence-2 model", gr.update(), "", ""
        
        # Update progress
        progress_html = make_progress_bar_html(50, 'Generating caption...')
        
        # Move model to GPU for inference
        florence_model.to(gpu)
        generated_prompt = generate_caption(Image.fromarray(input_image))
        
        # Move model back to CPU to free GPU memory
        florence_model.to(cpu)
        torch.cuda.empty_cache()
        
        # Clear progress indicators once done
        return generated_prompt, gr.update(), "", ""
        
    except Exception as e:
        print(f"Error generating caption: {e}")
        if florence_model is not None:
            florence_model.to(cpu)
        torch.cuda.empty_cache()
        return "Error generating caption. Please try again or enter prompt manually.", gr.update(), "", ""


css = make_progress_bar_css()
block = gr.Blocks(css=css, title="FramePack-F1").queue()
with block:
    gr.Markdown('# FramePack-F1')
    
    with gr.Tabs() as tabs:
        # Image-to-Video Tab
        with gr.TabItem("Image-to-Video"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(sources='upload', type="numpy", label="Image", height=320)
                    
                    with gr.Row():
                        img2vid_prompt = gr.Textbox(label="Prompt", value='')
                        caption_button = gr.Button(value="📝 Generate Caption", scale=0.15)
                    
                    img2vid_resolution = gr.Dropdown(
                        label="Resolution", 
                        choices=["512", "576", "640", "704", "768", "832", "896", "960", "1024"], 
                        value="640",
                        info="Higher resolutions require more VRAM and may slow down processing"
                    )

                    with gr.Row():
                        img2vid_start_button = gr.Button(value="Start Generation")
                        img2vid_end_button = gr.Button(value="End Generation", interactive=False)

                    with gr.Group():
                        img2vid_use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')

                        img2vid_n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)  # Not used
                        
                        with gr.Row(equal_height=True):
                            img2vid_seed = gr.Number(label="Seed", value=31337, precision=0, scale=4)
                            img2vid_random_seed_button = gr.Button(value="🎲 Random", scale=1)

                        img2vid_total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                        img2vid_latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
                        img2vid_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')

                        img2vid_cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                        img2vid_gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                        img2vid_rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

                        img2vid_gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")

                        img2vid_mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs.")

                with gr.Column():
                    img2vid_preview_image = gr.Image(label="Next Latents", height=200, visible=False)
                    img2vid_result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
                    img2vid_progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
                    img2vid_progress_bar = gr.HTML('', elem_classes='no-generating-animation')

        # Text-to-Video Tab
        with gr.TabItem("Text-to-Video"):
            with gr.Row():
                with gr.Column():
                    txt2vid_prompt = gr.Textbox(label="Text Prompt", value='', lines=3, placeholder="Enter a detailed description of the video you want to generate...")
                    
                    with gr.Row():
                        enhance_prompt_button = gr.Button(value="✨ Enhance Prompt")
                    
                    txt2vid_resolution = gr.Dropdown(
                        label="Resolution", 
                        choices=["512", "576", "640", "704", "768", "832", "896", "960", "1024"], 
                        value="640",
                        info="Higher resolutions require more VRAM and may slow down processing"
                    )

                    with gr.Row():
                        txt2vid_start_button = gr.Button(value="Start Generation")
                        txt2vid_end_button = gr.Button(value="End Generation", interactive=False)

                    with gr.Group():
                        txt2vid_use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')

                        txt2vid_n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)  # Not used
                        
                        with gr.Row(equal_height=True):
                            txt2vid_seed = gr.Number(label="Seed", value=42424, precision=0, scale=4)
                            txt2vid_random_seed_button = gr.Button(value="🎲 Random", scale=1)

                        txt2vid_total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                        txt2vid_latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
                        txt2vid_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')

                        txt2vid_cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                        txt2vid_gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                        txt2vid_rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

                        txt2vid_gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")

                        txt2vid_mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs.")

                with gr.Column():
                    txt2vid_preview_image = gr.Image(label="Next Latents", height=200, visible=False)
                    txt2vid_result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
                    txt2vid_progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
                    txt2vid_progress_bar = gr.HTML('', elem_classes='no-generating-animation')

    gr.HTML('<div style="text-align:center; margin-top:20px;">Share your results and find ideas at the <a href="https://x.com/search?q=framepack&f=live" target="_blank">FramePack Twitter (X) thread</a></div>')

    def generate_random_seed():
        return np.random.randint(0, 2**31 - 1)
    
    # Image-to-Video event handlers
    img2vid_random_seed_button.click(fn=generate_random_seed, inputs=[], outputs=[img2vid_seed])
    caption_button.click(fn=process_caption, inputs=[input_image], outputs=[img2vid_prompt, img2vid_preview_image, img2vid_progress_desc, img2vid_progress_bar])

    img2vid_ips = [input_image, img2vid_prompt, img2vid_n_prompt, img2vid_seed, img2vid_total_second_length, img2vid_latent_window_size, img2vid_steps, img2vid_cfg, img2vid_gs, img2vid_rs, img2vid_gpu_memory_preservation, img2vid_use_teacache, img2vid_mp4_crf, img2vid_resolution]
    img2vid_start_button.click(fn=process_image_to_video, inputs=img2vid_ips, outputs=[img2vid_result_video, img2vid_preview_image, img2vid_progress_desc, img2vid_progress_bar, img2vid_start_button, img2vid_end_button])
    img2vid_end_button.click(fn=end_process)
    
    # Text-to-Video event handlers
    txt2vid_random_seed_button.click(fn=generate_random_seed, inputs=[], outputs=[txt2vid_seed])
    enhance_prompt_button.click(fn=enhance_prompt_event, inputs=[txt2vid_prompt], outputs=[txt2vid_prompt])
    
    txt2vid_ips = [txt2vid_prompt, txt2vid_n_prompt, txt2vid_seed, txt2vid_total_second_length, txt2vid_latent_window_size, txt2vid_steps, txt2vid_cfg, txt2vid_gs, txt2vid_rs, txt2vid_gpu_memory_preservation, txt2vid_use_teacache, txt2vid_mp4_crf, txt2vid_resolution]
    txt2vid_start_button.click(fn=process_text_to_video, inputs=txt2vid_ips, outputs=[txt2vid_result_video, txt2vid_preview_image, txt2vid_progress_desc, txt2vid_progress_bar, txt2vid_start_button, txt2vid_end_button])
    txt2vid_end_button.click(fn=end_process)


block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)
