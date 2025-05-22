# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modified from diffusers==0.29.2
#
# ==============================================================================
import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch
import torch.distributed as dist
import numpy as np
from dataclasses import dataclass
from packaging import version

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput
import math

from ...constants import PRECISION_TO_TYPE
from ...vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from ...text_encoder import TextEncoder
from ...modules import HYVideoDiffusionTransformer
from ...utils.data_utils import black_image

import logging
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """"""

def get_dimension_slices_and_sizes(begin, end, size):

    slices = []
    sizes = [] 
    current_pos = begin
    
    while current_pos < end:
        start_idx = current_pos % size # Start index in the current tile
        next_boundary = ((current_pos // size) + 1) * size # The start position of the next tile
        end_pos = min(end, next_boundary) # The end position of the current tile
        length = end_pos - current_pos
        end_idx = (start_idx + length) % size # End index in the current tile

        if end_idx > start_idx:
            slices.append(slice(start_idx, end_idx))
            sizes.append(end_idx - start_idx)
        else: 
            slices.append(slice(start_idx, size))
            sizes.append(size - start_idx)
            if end_idx > 0:
                slices.append(slice(0, end_idx))
                sizes.append(end_idx)
        current_pos = end_pos
    
    return slices, sizes
        
            
class RingLatent2D:
    def __init__(self, latent_tensor):
        self.torch_latent = latent_tensor.clone()
        self.batch_size = self.torch_latent.shape[0] 
        self.num_frames = self.torch_latent.shape[1]
        self.channels = self.torch_latent.shape[2]
        self.height = self.torch_latent.shape[3]
        self.width = self.torch_latent.shape[4]

    def get_shape(self):
        return self.torch_latent.shape
    
    def get_window_latent(self, top: int = None, bottom: int = None, left: int = None, right: int = None):
        if top is None:
            top = 0
        if bottom is None:
            bottom = self.height
        if left is None:
            left = 0
        if right is None:
            right = self.width
            
        # Ensure the indices are within the valid range
        assert 0 <= top < bottom <= self.height * 2, f"Invalid top {top} and bottom {bottom}"
        assert 0 <= left < right <= self.width * 2, f"Invalid left {left} and right {right}"
        
        height_slices, height_sizes = get_dimension_slices_and_sizes(top, bottom, self.height)
        width_slices, width_sizes = get_dimension_slices_and_sizes(left, right, self.width)

        # Get the parts of the latent tensor
        parts = []
        for h_slice in height_slices:
            row_parts = []
            for w_slice in width_slices:
                part = self.torch_latent[:, :, :, h_slice, w_slice]
                row_parts.append(part)
            row = torch.cat(row_parts, dim=4)
            parts.append(row)
        desired_latent = torch.cat(parts, dim=3)
        
        return desired_latent.clone()
    
    def set_window_latent(self, input_latent: torch.Tensor,
                          top: int = None,
                          bottom: int = None,
                          left: int = None,
                          right: int = None):
        if top is None:
            top = 0
        if bottom is None:
            bottom = self.height
        if left is None:
            left = 0
        if right is None:
            right = self.width

        assert 0 <= top < bottom <= self.height * 2, f"Invalid top {top} and bottom {bottom}"
        assert 0 <= left < right <= self.width * 2, f"Invalid left {left} and right {right}"
        assert bottom - top <= self.height, f"warp should not occur"
        assert right - left <= self.width, f"warp should not occur"

       # Calculate the target latent tensor
        target_height = bottom - top if bottom <= self.height else (self.height - top) + (bottom % self.height)
        target_width = right - left if right <= self.width else (self.width - left) + (right % self.width)

        width_slices, width_sizes = get_dimension_slices_and_sizes(left, right, self.width)
        height_slices, height_sizes = get_dimension_slices_and_sizes(top, bottom, self.height)

        target_height = sum(height_sizes)
        target_width = sum(width_sizes)
        # Check the shape of the input latent tensor
        assert input_latent.shape[3:] == (target_height, target_width), f"Input latent shape {input_latent.shape[3:]} does not match target window shape {(target_height, target_width)}"

        # Write the parts of the latent tensor
        h_start = 0
        for h_slice, h_size in zip(height_slices, height_sizes):
            w_start = 0
            for w_slice, w_size in zip(width_slices, width_sizes):
                input_part = input_latent[:, :, :, h_start:h_start+h_size, w_start:w_start+w_size]
                self.torch_latent[:, :, :, h_slice, w_slice] = input_part
                w_start += w_size
            h_start += h_size

# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps



# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")



@dataclass
class HunyuanVideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class HunyuanVideoPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using HunyuanVideo.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`TextEncoder`]):
            Frozen text-encoder.
        text_encoder_2 ([`TextEncoder`]):
            Frozen text-encoder_2.
        transformer ([`HYVideoDiffusionTransformer`]):
            A `HYVideoDiffusionTransformer` to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = ["text_encoder_2"]
    _exclude_from_cpu_offload = ["transformer"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: TextEncoder,
        transformer: HYVideoDiffusionTransformer,
        scheduler: KarrasDiffusionSchedulers,
        text_encoder_2: Optional[TextEncoder] = None,
        progress_bar_config: Dict[str, Any] = None,
        args=None,
    ):
        super().__init__()

        # ==========================================================================================
        if progress_bar_config is None:
            progress_bar_config = {}
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(progress_bar_config)

        self.args = args
        # ==========================================================================================

        if (
            hasattr(scheduler.config, "steps_offset")
            and scheduler.config.steps_offset != 1
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate(
                "steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if (
            hasattr(scheduler.config, "clip_sample")
            and scheduler.config.clip_sample is True
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate(
                "clip_sample not set", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            text_encoder_2=text_encoder_2,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
        text_encoder: Optional[TextEncoder] = None,
        data_type: Optional[str] = "image",
        semantic_images=None
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_videos_per_prompt (`int`):
                number of videos that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            attention_mask (`torch.Tensor`, *optional*):
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_attention_mask (`torch.Tensor`, *optional*):
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            text_encoder (TextEncoder, *optional*):
            data_type (`str`, *optional*):
        """
        if text_encoder is None:
            text_encoder = self.text_encoder

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(text_encoder.model, lora_scale)
            else:
                scale_lora_layers(text_encoder.model, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, text_encoder.tokenizer)

            text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)

            if clip_skip is None:
                prompt_outputs = text_encoder.encode(
                    text_inputs, data_type=data_type, semantic_images=semantic_images, device=device
                )
                prompt_embeds = prompt_outputs.hidden_state
            else:
                prompt_outputs = text_encoder.encode(
                    text_inputs,
                    output_hidden_states=True,
                    data_type=data_type,
                    semantic_images=semantic_images,
                    device=device,
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_outputs.hidden_states_list[-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = text_encoder.model.text_model.final_layer_norm(
                    prompt_embeds
                )

            attention_mask = prompt_outputs.attention_mask
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                bs_embed, seq_len = attention_mask.shape
                attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
                attention_mask = attention_mask.view(
                    bs_embed * num_videos_per_prompt, seq_len
                )

        if text_encoder is not None:
            prompt_embeds_dtype = text_encoder.dtype
        elif self.transformer is not None:
            prompt_embeds_dtype = self.transformer.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        if prompt_embeds.ndim == 2:
            bs_embed, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
            prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
        else:
            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(
                bs_embed * num_videos_per_prompt, seq_len, -1
            )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(
                    uncond_tokens, text_encoder.tokenizer
                )

            # max_length = prompt_embeds.shape[1]
            uncond_input = text_encoder.text2tokens(uncond_tokens, data_type=data_type)

            if semantic_images is not None:
                uncond_image = [black_image(img.size[0], img.size[1]) for img in semantic_images]
            else:
                uncond_image = None

            negative_prompt_outputs = text_encoder.encode(
                uncond_input, data_type=data_type, semantic_images=uncond_image, device=device
            )
            negative_prompt_embeds = negative_prompt_outputs.hidden_state

            negative_attention_mask = negative_prompt_outputs.attention_mask
            if negative_attention_mask is not None:
                negative_attention_mask = negative_attention_mask.to(device)
                _, seq_len = negative_attention_mask.shape
                negative_attention_mask = negative_attention_mask.repeat(
                    1, num_videos_per_prompt
                )
                negative_attention_mask = negative_attention_mask.view(
                    batch_size * num_videos_per_prompt, seq_len
                )

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=prompt_embeds_dtype, device=device
            )

            if negative_prompt_embeds.ndim == 2:
                negative_prompt_embeds = negative_prompt_embeds.repeat(
                    1, num_videos_per_prompt
                )
                negative_prompt_embeds = negative_prompt_embeds.view(
                    batch_size * num_videos_per_prompt, -1
                )
            else:
                negative_prompt_embeds = negative_prompt_embeds.repeat(
                    1, num_videos_per_prompt, 1
                )
                negative_prompt_embeds = negative_prompt_embeds.view(
                    batch_size * num_videos_per_prompt, seq_len, -1
                )

        if text_encoder is not None:
            if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(text_encoder.model, lora_scale)

        return (
            prompt_embeds,
            negative_prompt_embeds,
            attention_mask,
            negative_attention_mask,
        )

    def decode_latents(self, latents, enable_tiling=True):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        if enable_tiling:
            self.vae.enable_tiling()
            image = self.vae.decode(latents, return_dict=False)[0]
        else:
            image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        if image.ndim == 4:
            image = image.cpu().permute(0, 2, 3, 1).float()
        else:
            image = image.cpu().float()
        return image

    def prepare_extra_func_kwargs(self, func, kwargs):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        extra_step_kwargs = {}

        for k, v in kwargs.items():
            accepts = k in set(inspect.signature(func).parameters.keys())
            if accepts:
                extra_step_kwargs[k] = v
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        video_length,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        vae_ver="88-4c-sd",
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if video_length is not None:
            if "884" in vae_ver:
                if video_length != 1 and (video_length - 1) % 4 != 0:
                    raise ValueError(
                        f"`video_length` has to be 1 or a multiple of 4 but is {video_length}."
                    )
            elif "888" in vae_ver:
                if video_length != 1 and (video_length - 1) % 8 != 0:
                    raise ValueError(
                        f"`video_length` has to be 1 or a multiple of 8 but is {video_length}."
                    )

        if callback_steps is not None and (
            not isinstance(callback_steps, int) or callback_steps <= 0
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs
            for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents, 
        height, 
        width,  
        video_length, # Full video latent frame count
        dtype,
        device, 
        generator, 
        latents: Optional[torch.Tensor] = None, # Optional external x_T
        img_latents: Optional[torch.Tensor] = None, 
        i2v_mode=False,
        i2v_condition_type=None,
        i2v_stability=True,
    ):
        shape_num_channels = self.vae.config.latent_channels
        if i2v_mode and i2v_condition_type == "latent_concat":
            pass


        shape = (
            batch_size, shape_num_channels, video_length,
            int(height) // self.vae_scale_factor, int(width) // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"Batch size must match length of generators list.")

        creation_device = device
        if isinstance(generator, torch.Generator):
            creation_device = generator.device
        elif isinstance(generator, list) and len(generator) > 0:
            creation_device = generator[0].device
        
        prepared_latents: torch.Tensor 
        if i2v_mode and i2v_stability:
            assert img_latents is not None and img_latents.shape[1] == shape_num_channels, \
                "img_latents required for i2v_stability with matching VAE channels."
            
            img_latents_on_creation_device = img_latents.to(creation_device)
            img_latents_expanded_time = img_latents_on_creation_device.repeat(1, 1, video_length, 1, 1) if img_latents_on_creation_device.shape[2] == 1 else img_latents_on_creation_device
            
            x0 = randn_tensor(shape, generator=generator, device=creation_device, dtype=dtype)
            t_stability = torch.tensor([0.999], device=creation_device, dtype=dtype) 
            prepared_latents = x0 * t_stability + img_latents_expanded_time * (1 - t_stability)
            prepared_latents = prepared_latents.to(dtype=dtype) 
        
        elif latents is None: 
            prepared_latents = randn_tensor(shape, generator=generator, device=creation_device, dtype=dtype)
        else: 
            prepared_latents = latents.to(creation_device, dtype=dtype) 
            if prepared_latents.shape != shape:
                raise ValueError(f"Provided latents shape {prepared_latents.shape} doesn't match {shape}.")

        prepared_latents = prepared_latents.to(device)

        if hasattr(self.scheduler, "init_noise_sigma"):
            prepared_latents = prepared_latents * self.scheduler.init_noise_sigma
        return prepared_latents

    def get_1d_rotary_pos_embed_riflex(
        self, dim: int, pos: Union[np.ndarray, int], theta: float = 10000.0,
        use_real=False, k: Optional[int] = None, L_test: Optional[int] = None,
    ):
        assert dim % 2 == 0
        # Ensure pos is a tensor on the correct device.
        if isinstance(pos, int): pos = torch.arange(pos, device=self.device)
        if isinstance(pos, np.ndarray): pos = torch.from_numpy(pos).to(self.device)

        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=pos.device)[: (dim // 2)].float() / dim))
        if k is not None and L_test is not None and L_test > 0: # RIFLEx modification.
            freqs[k-1] = 0.9 * 2 * torch.pi / L_test
        
        freqs = torch.outer(pos, freqs)
        if use_real: # Return separate cos and sin components.
            return freqs.cos().repeat_interleave(2, dim=1).float(), freqs.sin().repeat_interleave(2, dim=1).float()
        else: # Return complex polar representation.
            return torch.polar(torch.ones_like(freqs), freqs)

    def get_rotary_pos_embed(self, video_length: int, window_pixel_h: int, window_pixel_w: int): # video_length is pixel frames
        target_ndim = 3 # T, H, W for RoPE

        # Calculate latent dimensions for RoPE.
        if "884" in self.args.vae:
            latent_temporal_dim = (video_length - 1) // 4 + 1
        elif "888" in self.args.vae:
            latent_temporal_dim = (video_length - 1) // 8 + 1
        else:
            latent_temporal_dim = video_length # Assuming 1-to-1 pixel to latent frame if not 884/888
        
        window_latent_h = window_pixel_h // self.vae_scale_factor
        window_latent_w = window_pixel_w // self.vae_scale_factor
        current_rope_latent_dims = [latent_temporal_dim, window_latent_h, window_latent_w]

        # Determine patch sizes (temporal, height, width).
        if isinstance(self.transformer.config.patch_size, int):
            p_t, p_h, p_w = [self.transformer.config.patch_size] * 3
        elif isinstance(self.transformer.config.patch_size, list) and len(self.transformer.config.patch_size) == 3:
            p_t, p_h, p_w = self.transformer.config.patch_size
        else:
            raise ValueError("Invalid transformer.config.patch_size format.")

        # Calculate number of patches (rope_sizes).
        assert all(current_rope_latent_dims[i] % [p_t, p_h, p_w][i] == 0 for i in range(3)), \
            f"Latent dims {current_rope_latent_dims} not divisible by patch sizes {[p_t, p_h, p_w]}."
        rope_sizes = [current_rope_latent_dims[i] // [p_t, p_h, p_w][i] for i in range(3)]
            
        L_test = rope_sizes[0] 
        L_train = 25 
    
        head_dim = self.transformer.config.hidden_size // self.transformer.config.heads_num
        rope_dim_list = self.transformer.config.rope_dim_list or [head_dim // target_ndim] * target_ndim
        assert sum(rope_dim_list) == head_dim, "Sum of rope_dim_list must equal head_dim."
    
        k_param_riflex = None 
        if video_length > 192: 
            k_param_riflex = 2 + ((video_length + 3) // (4 * L_train))
            k_param_riflex = max(4, min(8, k_param_riflex))
            logger.debug(f"RIFLEx active: k={k_param_riflex} for {video_length} frames.")
        else:
            logger.debug(f"Standard RoPE for {video_length} frames.")

        axes_grids = [torch.arange(s, device=self.device, dtype=torch.float32) for s in rope_sizes]
        grid = torch.meshgrid(*axes_grids, indexing="ij")
        pos = torch.stack(grid, dim=0).reshape(3, -1).t() 

        freqs_list = [] 
        for i_dim in range(3): 
            current_k_for_riflex = k_param_riflex if i_dim == 0 and video_length > 192 else None
            current_L_test_for_riflex = L_test if i_dim == 0 and video_length > 192 else None
            
            cos_d, sin_d = self.get_1d_rotary_pos_embed_riflex(
                rope_dim_list[i_dim], 
                pos[:, i_dim], 
                theta=self.args.rope_theta,
                use_real=True,
                k=current_k_for_riflex, 
                L_test=current_L_test_for_riflex
            )
            freqs_list.append((cos_d, sin_d))
        
        freqs_cos = torch.cat([f[0] for f in freqs_list], dim=1)
        freqs_sin = torch.cat([f[1] for f in freqs_list], dim=1)
        
        return freqs_cos.to(self.device), freqs_sin.to(self.device)

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(
        self,
        w: torch.Tensor,
        embedding_dim: int = 512,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        # return self._guidance_scale > 1 and self.transformer.config.time_cond_proj_dim is None
        return self._guidance_scale > 1

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        video_length: int, 
        data_type: str = "video",
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None, 
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None, 
        vae_ver: str = "88-4c-sd",
        enable_tiling: bool = False,
        n_tokens: Optional[int] = None, 
        embedded_guidance_scale: Optional[float] = None,
        i2v_mode: bool = False,
        i2v_condition_type: str = None,
        i2v_stability: bool = True,
        img_latents: Optional[torch.Tensor] = None,
        semantic_images=None,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`):
                The height in pixels of the generated image.
            width (`int`):
                The width in pixels of the generated image.
            video_length (`int`):
                The number of frames in the generated video.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
                
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`HunyuanVideoPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~HunyuanVideoPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`HunyuanVideoPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        # height = height or self.transformer.config.sample_size * self.vae_scale_factor
        # width = width or self.transformer.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            video_length, # Pixel video length
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            vae_ver=vae_ver,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = torch.device(f"cuda:{dist.get_rank()}") if dist.is_initialized() else self._execution_device

        # Ensure transformer is on the correct device
        self.transformer.to(device)

        # Ensure img_latents (if provided for i2v_mode) is on the correct device and not meta
        if i2v_mode and img_latents is not None:
            if img_latents.device.type == 'meta' or img_latents.device != device:
                logger.debug(f"Moving img_latents from {img_latents.device} to {device}")
                img_latents = img_latents.to(device)

        original_video_frames = video_length # Store original pixel video length

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_mask,
            negative_prompt_mask,
        ) = self.encode_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            attention_mask=attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_attention_mask=negative_attention_mask,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
            data_type=data_type,
            semantic_images=semantic_images
        )
        if self.text_encoder_2 is not None:
            (
                prompt_embeds_2,
                negative_prompt_embeds_2,
                prompt_mask_2,
                negative_prompt_mask_2,
            ) = self.encode_prompt(
                prompt,
                device,
                num_videos_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=None,
                attention_mask=None,
                negative_prompt_embeds=None,
                negative_attention_mask=None,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
                text_encoder=self.text_encoder_2,
                data_type=data_type,
            )
        else:
            prompt_embeds_2 = None
            negative_prompt_embeds_2 = None
            prompt_mask_2 = None
            negative_prompt_mask_2 = None

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_mask is not None:
                prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])
            if prompt_embeds_2 is not None:
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
            if prompt_mask_2 is not None:
                prompt_mask_2 = torch.cat([negative_prompt_mask_2, prompt_mask_2])

        # 4. Prepare timesteps
        scheduler_n_tokens = n_tokens 
        extra_set_timesteps_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.set_timesteps, {"n_tokens": scheduler_n_tokens}
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            self.device,
            timesteps,
            sigmas,
            **extra_set_timesteps_kwargs,
        )

        # Convert video_length from pixel frames to latent frames for prepare_latents
        latent_video_length = original_video_frames 
        if "884" in vae_ver:
            latent_video_length = (latent_video_length - 1) // 4 + 1
        elif "888" in vae_ver:
            latent_video_length = (latent_video_length - 1) // 8 + 1
        # else: latent_video_length remains original_video_frames (pixel frames)

        # 5. Prepare initial latents (x_T)
        # `latents` here is the initial optional latents passed to __call__
        transformer_in_channels = self.transformer.config.in_channels
        initial_latents_xt = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            transformer_in_channels, 
            height,
            width,
            latent_video_length, # Pass latent_video_length
            prompt_embeds.dtype,
            self.device, # This is self._execution_device
            generator, 
            latents, # Pass external latents if provided
            img_latents, # img_latents is already on the target CUDA `device` due to earlier .to(device)
            i2v_mode, 
            i2v_condition_type, 
            i2v_stability)
        
        # Explicitly move initial_latents_xt to the target computation `device` (CUDA)
        initial_latents_xt = initial_latents_xt.to(device)
        
        num_denoised_latent_channels = initial_latents_xt.shape[1] 

        img_latents_cond_full, mask_concat_full = None, None
        if i2v_mode and i2v_condition_type == "latent_concat":
            assert img_latents is not None and img_latents.shape[1] == num_denoised_latent_channels, \
                "img_latents (VAE encoded) required for latent_concat with matching channels."
            # Ensure img_latents_cond_full has latent_video_length frames for RingLatent2D
            img_latents_cond_full = img_latents.repeat(1,1,latent_video_length,1,1) if img_latents.shape[2] == 1 else img_latents
            if img_latents_cond_full.shape[2] != latent_video_length: # If original img_latents had >1 frame but not matching
                 img_latents_cond_full = img_latents_cond_full[:,:,:latent_video_length,:,:] 
                 logger.warning(f"img_latents for concat was reshaped to {latent_video_length} frames")

            # Ensure mask_concat_full is created on the target CUDA `device`
            mask_concat_full = torch.ones(initial_latents_xt.shape[0], 1, latent_video_length, 
                                          initial_latents_xt.shape[3], initial_latents_xt.shape[4], 
                                          device=device, dtype=prompt_embeds.dtype)
            if latent_video_length > 1: mask_concat_full[:, :, 1:, ...] = 0

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step,
            {"generator": generator, "eta": eta},
        )

        target_dtype = PRECISION_TO_TYPE[self.args.precision]
        autocast_enabled = (
            target_dtype != torch.float32
        ) and not self.args.disable_autocast
        vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not self.args.disable_autocast

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        ###### Define the sliding-window related params ######
        ll_height, ll_width = initial_latents_xt.shape[3], initial_latents_xt.shape[4] 
        window_h_latent = ll_height 
        window_w_latent = max(ll_width // 2, self.transformer.config.patch_size[2] if isinstance(self.transformer.config.patch_size, list) else self.transformer.config.patch_size)
        window_size_latent = (window_h_latent, window_w_latent)

        num_windows_h = math.ceil(ll_height / window_size_latent[0])
        num_windows_w = math.ceil(ll_width / window_size_latent[1])
        loop_step = 8 
        latent_step_size_w = window_size_latent[1] // loop_step if num_windows_w > 1 else 0 
        latent_step_size_h = window_size_latent[0] // loop_step if num_windows_h > 1 else 0 
        
        # initial_latents_xt is now on the target CUDA `device`
        ring_latent_handler = RingLatent2D(initial_latents_xt) 
        # Ensure aggregated_noise_handler's internal tensor is on the target CUDA `device`
        aggregated_noise_handler = RingLatent2D(torch.zeros_like(initial_latents_xt, device=device)) 
        
        ring_image_latent_handler = None
        if i2v_mode and img_latents_cond_full is not None: # img_latents_cond_full is on CUDA `device`
            ring_image_latent_handler = RingLatent2D(img_latents_cond_full)
        elif i2v_mode and img_latents is not None: # Fallback if cond_full wasn't prepared but img_latents exists
             # This path might need img_latents to be correctly tiled to latent_video_length for RingLatent2D
             tiled_img_latents = img_latents.repeat(1,1,latent_video_length,1,1) if img_latents.shape[2] == 1 else img_latents
             if tiled_img_latents.shape[2] != latent_video_length:
                  tiled_img_latents = tiled_img_latents[:,:,:latent_video_length,:,:] # Basic truncate
             ring_image_latent_handler = RingLatent2D(tiled_img_latents)


        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps): 
                if self.interrupt: continue
                current_full_latents_xt = ring_latent_handler.torch_latent.clone() 
                aggregated_noise_handler.torch_latent.zero_() 
                noise_contribution_counts = torch.zeros_like(aggregated_noise_handler.torch_latent, dtype=torch.int, device=device) # Ensure on CUDA

                current_grid_offset_left = (i % loop_step) * latent_step_size_w
                current_grid_offset_top = (i % loop_step) * latent_step_size_h
                
                for shift_h_idx in range(num_windows_h): 
                    for shift_w_idx in range(num_windows_w): 
                        window_latent_top = current_grid_offset_top + shift_h_idx * window_size_latent[0] 
                        window_latent_down = window_latent_top + window_size_latent[0] 
                        window_latent_left = current_grid_offset_left + shift_w_idx * window_size_latent[1] 
                        window_latent_right = window_latent_left + window_size_latent[1] 
                        
                        latents_for_view = ring_latent_handler.get_window_latent(window_latent_top, window_latent_down, window_latent_left, window_latent_right) 
                        if latents_for_view.shape[3] == 0 or latents_for_view.shape[4] == 0: continue

                        current_window_pixel_height = latents_for_view.shape[3] * self.vae_scale_factor 
                        current_window_pixel_width = latents_for_view.shape[4] * self.vae_scale_factor 
                        window_freqs_cos, window_freqs_sin = self.get_rotary_pos_embed(original_video_frames, current_window_pixel_height, current_window_pixel_width) 
                        # Ensure rotary embeddings are on the correct device
                        window_freqs_cos = window_freqs_cos.to(device)
                        window_freqs_sin = window_freqs_sin.to(device)
                        
                        latent_model_input = latents_for_view.to(device) # Ensure latents_for_view is on the target device
                        if i2v_mode:
                            image_latents_for_view = ring_image_latent_handler.get_window_latent(window_latent_top, window_latent_down, window_latent_left, window_latent_right) if ring_image_latent_handler else None
                            if image_latents_for_view is not None: 
                                image_latents_for_view = image_latents_for_view.to(device) # Ensure device consistency
                                if i2v_condition_type == "token_replace":
                                    # first_cond_f will be on `device` because image_latents_for_view is
                                    first_cond_f = image_latents_for_view[:,:,0:1,:,:] if image_latents_for_view.shape[2] > 0 else image_latents_for_view
                                    
                                    # latents_for_view_slice will be on `device` because latent_model_input (from latents_for_view) is
                                    latents_for_view_slice = latent_model_input[:,:,1:,:,:]
                                    
                                    if latents_for_view.shape[2] > 1:
                                        latent_model_input = torch.cat([first_cond_f, latents_for_view_slice], dim=2) 
                                    elif latents_for_view.shape[2] == 1 and first_cond_f.shape[2] == 1 : 
                                        latent_model_input = first_cond_f 
                                    else: 
                                        latent_model_input = first_cond_f 
                                elif i2v_condition_type == "latent_concat":
                                    # Ensure mask_concat_full is on the correct device before RingLatent2D
                                    # and image_latents_for_view is already on device
                                    mask_view = RingLatent2D(mask_concat_full).get_window_latent(window_latent_top,window_latent_down,window_latent_left,window_latent_right)
                                    latent_model_input = torch.cat([latent_model_input, image_latents_for_view, mask_view.to(device)], dim=1)
                        
                        input_cfg = torch.cat([latent_model_input.to(device)]*2) if self.do_classifier_free_guidance else latent_model_input.to(device)
                        
                        t_on_device = t.to(device) if isinstance(t, torch.Tensor) else torch.tensor(t, device=device, dtype=torch.long) # Ensure t is long for scheduler
                        input_scaled = self.scheduler.scale_model_input(input_cfg, t_on_device)
                        t_exp = t_on_device.repeat(input_scaled.shape[0])
                        guid_p = (torch.tensor([embedded_guidance_scale]*input_scaled.shape[0],dtype=target_dtype,device=device)*1000.0) if embedded_guidance_scale else None

                        with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):
                            noise_raw = self.transformer(input_scaled, t_exp, prompt_embeds, prompt_mask, 
                                                         prompt_embeds_2, window_freqs_cos, window_freqs_sin, guid_p, True)["x"] 
                        
                        noise_pred_uncond, noise_pred_text = None, None 
                        if self.do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_raw.chunk(2)
                            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond) 
                            if self.guidance_rescale > 0.0: noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, self.guidance_rescale)
                        else: noise_pred = noise_raw
                        
                        final_noise_win = noise_pred[:, :num_denoised_latent_channels, ...]
                        
                        noise_to_agg = torch.zeros_like(latents_for_view)
                        if i2v_mode and i2v_condition_type == "token_replace":
                            if final_noise_win.shape[2] == latents_for_view.shape[2] -1 and latents_for_view.shape[2] > 0: 
                                noise_to_agg[:,:,1:,:,:] = final_noise_win
                            elif final_noise_win.shape[2] == latents_for_view.shape[2]: 
                                noise_to_agg = final_noise_win
                        else: noise_to_agg = final_noise_win

                        h_slices, _ = get_dimension_slices_and_sizes(window_latent_top, window_latent_down, ll_height)
                        w_slices, _ = get_dimension_slices_and_sizes(window_latent_left, window_latent_right, ll_width)
                        h_off_v = 0
                        for hs_a in h_slices:
                            w_off_v = 0
                            hs_v = slice(h_off_v, h_off_v + (hs_a.stop - hs_a.start))
                            for ws_a in w_slices:
                                ws_v = slice(w_off_v, w_off_v + (ws_a.stop - ws_a.start))
                                aggregated_noise_handler.torch_latent[:,:,:,hs_a,ws_a] += noise_to_agg[:,:,:,hs_v,ws_v]
                                noise_contribution_counts[:,:,:,hs_a,ws_a] += 1
                                w_off_v += (ws_a.stop - ws_a.start)
                            h_off_v += (hs_a.stop - hs_a.start)
                valid_counts = noise_contribution_counts.float().clamp(min=1.0)
                avg_model_out = aggregated_noise_handler.torch_latent / valid_counts
                
                model_out_sched = avg_model_out
                sample_sched = current_full_latents_xt
                if i2v_mode and i2v_condition_type == "token_replace": 
                    if avg_model_out.shape[2] > 0 and current_full_latents_xt.shape[2] > 0 and \
                       avg_model_out.shape[2] == current_full_latents_xt.shape[2] -1 : # Check if model output is one frame shorter
                        model_out_sched = avg_model_out # Already F-1
                        sample_sched = current_full_latents_xt[:,:,1:,:,:] # x_t also F-1
                    elif avg_model_out.shape[2] == current_full_latents_xt.shape[2] and avg_model_out.shape[2] > 0: # If model output full length for some reason
                        model_out_sched = avg_model_out[:,:,1:,:,:]
                        sample_sched = current_full_latents_xt[:,:,1:,:,:]


                denoised_part = self.scheduler.step(model_out_sched, t, sample_sched, **extra_step_kwargs)[0]

                denoised_next_t = None
                if i2v_mode and i2v_condition_type == "token_replace":
                    if current_full_latents_xt.shape[2] > 0: 
                        first_frame_to_keep = current_full_latents_xt[:,:,0:1,:,:] 
                        if first_frame_to_keep.shape[2] + denoised_part.shape[2] == current_full_latents_xt.shape[2]:
                            denoised_next_t = torch.cat([first_frame_to_keep, denoised_part], dim=2)
                        else: 
                            denoised_next_t = denoised_part 
                            logger.warning(f"Frame concat issue in token_replace. Result frames: {denoised_part.shape[2]}, expected {current_full_latents_xt.shape[2]-1 if current_full_latents_xt.shape[2]>0 else 0}")
                    else: 
                        denoised_next_t = denoised_part
                else: 
                    denoised_next_t = denoised_part
                
                ring_latent_handler.torch_latent = denoised_next_t 

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k_cb in callback_on_step_end_tensor_inputs: 
                        callback_kwargs[k_cb] = locals()[k_cb]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    _latents_from_cb = callback_outputs.pop("latents", None)
                    if _latents_from_cb is not None:
                        latents = _latents_from_cb 
                        
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    if progress_bar is not None:
                        progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, ring_latent_handler.torch_latent) 

        final_latents = ring_latent_handler.torch_latent.clone() 

        if not output_type == "latent":
            expand_temporal_dim = False
            if len(final_latents.shape) == 4: 
                if isinstance(self.vae, AutoencoderKLCausal3D):
                    final_latents = final_latents.unsqueeze(2) 
                    expand_temporal_dim = True
            elif len(final_latents.shape) == 5: 
                pass
            else:
                raise ValueError(
                    f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {final_latents.shape}." 
                )

            if (
                hasattr(self.vae.config, "shift_factor")
                and self.vae.config.shift_factor
            ):
                image_latents = ( 
                    final_latents / self.vae.config.scaling_factor 
                    + self.vae.config.shift_factor
                )
            else:
                image_latents = final_latents / self.vae.config.scaling_factor 

            with torch.autocast(
                device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled
            ):
                if enable_tiling:
                    self.vae.enable_tiling()
                    image = self.vae.decode(
                        image_latents, return_dict=False, generator=generator 
                    )[0]
                else:
                    image = self.vae.decode(
                        image_latents, return_dict=False, generator=generator 
                    )[0]

            if expand_temporal_dim or image.shape[2] == 1:
                image = image.squeeze(2)
        else:
            image = final_latents

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().float()

        if i2v_mode and i2v_condition_type == "latent_concat":
            if image.shape[2] > 4: 
                image = image[:, :, 4:, :, :]
            else:
                logger.warning(f"i2v_mode with latent_concat: frame slicing expects >4 frames, got {image.shape[2]}. Skipping slice.")

        self.maybe_free_model_hooks()

        if not return_dict:
            return image

        return HunyuanVideoPipelineOutput(videos=image)