# Derived from zero123-hf.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import kornia
import numpy as np
import PIL
import torch
from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.configuration_utils import ConfigMixin, FrozenDict, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPFeatureExtractor, CLIPVisionModelWithProjection

from svdrr_transformer_2d import SvdrrTransformer2DModel


# --- Logger --- #
logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        >>> from pipeline_svdrr_DiT import SvdrrDiTPipeline
        >>> pipe = SvdrrDiTPipeline.from_pretrained(...)
        >>> image = pipe(input_imgs=img, poses=pose).images[0]
"""


class CCProjection(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, in_channel=772, out_channel=1152):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.projection = torch.nn.Linear(in_channel, out_channel)

    def forward(self, x):
        return self.projection(x)


class SvdrrDiTPipeline(DiffusionPipeline):
    """Pipeline for single-view conditioned novel view generation using SV-DRR."""

    _optional_components = ["safety_checker", "feature_extractor"]
    model_cpu_offload_seq = "image_encoder->cc_projection->transformer->vae"

    transformer: SvdrrTransformer2DModel

    def __init__(
        self,
        vae: AutoencoderKL,
        image_encoder: CLIPVisionModelWithProjection,
        transformer: SvdrrTransformer2DModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        cc_projection: CCProjection,
        requires_safety_checker: bool = False,
    ):
        super().__init__()
        self.transformer = transformer
        if (
            hasattr(scheduler.config, "steps_offset")
            and scheduler.config.steps_offset != 1
        ):
            deprecation_message = (
                f"Outdated scheduler config for {scheduler}: `steps_offset` should be 1."
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
                f"Scheduler {scheduler} config `clip_sample` should be False."
            )
            deprecate(
                "clip_sample not set", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"Safety checker disabled for {self.__class__}."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Must define feature_extractor when safety_checker is used."
            )

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            transformer=transformer,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            cc_projection=cc_projection,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)
        self.stable_zero123 = False

    def enable_vae_slicing(self):
        """Enable sliced VAE decoding to save memory."""
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        """Disable sliced VAE decoding."""
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        """Enable tiled VAE decoding for processing larger images."""
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        """Disable tiled VAE decoding."""
        self.vae.disable_tiling()

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        """Encode prompt into text encoder hidden states."""
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(f"Input truncated by CLIP tokenizer: {removed_text}")

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` type mismatch: {type(negative_prompt)} != {type(prompt)}"
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError("Batch size mismatch for `negative_prompt`.")
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )[0]

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=self.text_encoder.dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def CLIP_preprocess(self, x):
        """Resize and normalize input image for OpenAI CLIP encoder."""
        dtype = x.dtype
        if isinstance(x, torch.Tensor):
            if x.min() < -1.0 or x.max() > 1.0:
                raise ValueError("Expected input tensor values in [-1, 1]")
        x = kornia.geometry.resize(
            x.to(torch.float32),
            (224, 224),
            interpolation="bicubic",
            align_corners=True,
            antialias=False,
        ).to(dtype=dtype)
        x = (x + 1.0) / 2.0
        x = kornia.enhance.normalize(
            x,
            torch.Tensor([0.48145466, 0.4578275, 0.40821073]),
            torch.Tensor([0.26862954, 0.26130258, 0.27577711]),
        )
        return x

    def _encode_image(
        self, image, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        """Extract CLIP image embeddings."""
        dtype = next(self.image_encoder.parameters()).dtype
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` must be Tensor, PIL Image, or list, got {type(image)}"
            )

        if isinstance(image, torch.Tensor):
            if image.ndim == 3:
                assert image.shape[0] == 3, "Image should be of shape (3, H, W)"
                image = image.unsqueeze(0)
            assert image.ndim == 4, "Image must have 4 dimensions"
            if image.min() < -1 or image.max() > 1:
                raise ValueError("Image should be in range [-1, 1]")
        else:
            if isinstance(image, (PIL.Image.Image, np.ndarray)):
                image = [image]

            if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
                image = [np.array(i.convert("RGB"))[None, :] for i in image]
                image = np.concatenate(image, axis=0)
            elif isinstance(image, list) and isinstance(image[0], np.ndarray):
                image = np.concatenate([i[None, :] for i in image], axis=0)

            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        image = image.to(device=device, dtype=dtype)
        image = self.CLIP_preprocess(image)
        image_embeddings = self.image_encoder(image).image_embeds.to(dtype=dtype)
        image_embeddings = image_embeddings.unsqueeze(1)

        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
        image_embeddings = image_embeddings.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        if do_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(image_embeddings)
            image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

        return image_embeddings

    def _encode_pose(
        self, pose, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        """Encode camera pose into positional embeddings."""
        dtype = next(self.cc_projection.parameters()).dtype
        if isinstance(pose, torch.Tensor):
            pose_embeddings = pose.unsqueeze(1).to(device=device, dtype=dtype)
        else:
            if isinstance(pose[0], list):
                pose = torch.Tensor(pose)
            else:
                pose = torch.Tensor(np.array([pose]))
            x, y, z = (
                pose[:, 0].unsqueeze(dim=1),
                pose[:, 1].unsqueeze(1),
                pose[:, 2].unsqueeze(1),
            )
            if self.stable_zero123:
                z = torch.deg2rad(torch.zeros_like(z) + 90.0)
            pose_embeddings = (
                torch.cat(
                    [
                        torch.deg2rad(x),
                        torch.sin(torch.deg2rad(y)),
                        torch.cos(torch.deg2rad(y)),
                        z,
                    ],
                    dim=-1,
                )
                .unsqueeze(1)
                .to(device=device, dtype=dtype)
            )
        bs_embed, seq_len, _ = pose_embeddings.shape
        pose_embeddings = pose_embeddings.repeat(1, num_images_per_prompt, 1)
        pose_embeddings = pose_embeddings.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )
        if do_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(pose_embeddings)
            pose_embeddings = torch.cat([negative_prompt_embeds, pose_embeddings])
        return pose_embeddings

    def _encode_image_with_pose(
        self, image, pose, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        """Combine image and pose embeddings through cross-attention projection."""
        img_prompt_embeds = self._encode_image(
            image, device, num_images_per_prompt, False
        )
        pose_prompt_embeds = self._encode_pose(
            pose, device, num_images_per_prompt, False
        )
        prompt_embeds = torch.cat([img_prompt_embeds, pose_prompt_embeds], dim=-1)
        prompt_embeds = self.cc_projection(prompt_embeds)
        if do_classifier_free_guidance:
            negative_prompt = torch.zeros_like(prompt_embeds)
            prompt_embeds = torch.cat([negative_prompt, prompt_embeds])
        return prompt_embeds

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(
                self.numpy_to_pil(image), return_tensors="pt"
            ).to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    def decode_latents(self, latents):
        """Decode VAE latents into RGB image array."""
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, image, height, width, callback_steps):
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(f"Invalid image type: {type(image)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` must be divisible by 8: {height}x{width}."
            )

        if (callback_steps is None) or (
            callback_steps is not None
            and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(f"Invalid `callback_steps`: {callback_steps}")

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"Generators length {len(generator)} != batch size {batch_size}")

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_img_latents(
        self,
        image,
        batch_size,
        dtype,
        device,
        generator=None,
        do_classifier_free_guidance=False,
    ):
        """Encode input image into VAE latents with proper scaling factor."""
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(f"Invalid image type: {type(image)}")

        if isinstance(image, torch.Tensor):
            if image.ndim == 3:
                assert image.shape[0] == 3, "Image shape should be (3, H, W)"
                image = image.unsqueeze(0)
            assert image.ndim == 4, "Image must have 4 dimensions"
            if image.min() < -1 or image.max() > 1:
                raise ValueError("Image should be in [-1, 1] range")
        else:
            if isinstance(image, (PIL.Image.Image, np.ndarray)):
                image = [image]

            if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
                image = [np.array(i.convert("RGB"))[None, :] for i in image]
                image = np.concatenate(image, axis=0)
            elif isinstance(image, list) and isinstance(image[0], np.ndarray):
                image = np.concatenate([i[None, :] for i in image], axis=0)

            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        image = image.to(device=device, dtype=dtype)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"Generators length {len(generator)} != batch size {batch_size}")

        if isinstance(generator, list):
            init_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.mode(generator[i])
                for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = self.vae.encode(image).latent_dist.mode()

        # Scale input latents by VAE scaling factor to match training domain distribution
        init_latents = self.vae.config.scaling_factor * init_latents
        if batch_size > init_latents.shape[0]:
            num_images_per_prompt = batch_size // init_latents.shape[0]
            bs_embed, emb_c, emb_h, emb_w = init_latents.shape
            init_latents = init_latents.unsqueeze(1)
            init_latents = init_latents.repeat(1, num_images_per_prompt, 1, 1, 1)
            init_latents = init_latents.view(
                bs_embed * num_images_per_prompt, emb_c, emb_h, emb_w
            )

        init_latents = (
            torch.cat([torch.zeros_like(init_latents), init_latents])
            if do_classifier_free_guidance
            else init_latents
        )

        init_latents = init_latents.to(device=device, dtype=dtype)
        return init_latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        input_imgs: Union[torch.FloatTensor, PIL.Image.Image] = None,
        prompt_imgs: Union[torch.FloatTensor, PIL.Image.Image] = None,
        poses: Union[List[float], List[List[float]]] = None,
        torch_dtype=torch.float32,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: float = 1.0,
    ):
        """
        Pipeline generation call.

        Examples:

        Returns:
            StableDiffusionPipelineOutput or tuple
        """
        height = height or self.transformer.config.sample_size * self.vae_scale_factor
        width = width or self.transformer.config.sample_size * self.vae_scale_factor

        self.check_inputs(input_imgs, height, width, callback_steps)

        if isinstance(input_imgs, PIL.Image.Image):
            batch_size = 1
        elif isinstance(input_imgs, list):
            batch_size = len(input_imgs)
        else:
            batch_size = input_imgs.shape[0]
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds = self._encode_image_with_pose(
            prompt_imgs,
            poses,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            4,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        img_latents = self.prepare_img_latents(
            input_imgs,
            batch_size * num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        added_cond_kwargs = {
            "resolution": torch.tensor([height, width], device=device),
            "aspect_ratio": 1,
        }

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
                if self.transformer.config.in_channels == 8:
                    latent_model_input = torch.cat([latent_model_input, img_latents], dim=1)

                current_timestep = t
                if not torch.is_tensor(current_timestep):
                    is_mps = latent_model_input.device.type == "mps"
                    dtype = torch.float32 if is_mps else (torch.float64 if isinstance(current_timestep, float) else torch.int64)
                    current_timestep = torch.tensor(
                        [current_timestep],
                        dtype=dtype,
                        device=latent_model_input.device,
                    )
                elif len(current_timestep.shape) == 0:
                    current_timestep = current_timestep[None].to(
                        latent_model_input.device
                    )
                current_timestep = current_timestep.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=current_timestep,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                noise_pred = noise_pred.chunk(2, dim=1)[0]
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        has_nsfw_concept = None
        if output_type == "latent":
            image = latents
        elif output_type == "pil":
            image = self.decode_latents(latents)
            image = self.numpy_to_pil(image)
        else:
            image = self.decode_latents(latents)

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )

