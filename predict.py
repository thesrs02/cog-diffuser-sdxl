import os
from typing import List

import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionPipeline,
    EulerDiscreteScheduler,DPMSolverSDEScheduler,DEISMultistepScheduler,UniPCMultistepScheduler,KDPM2AncestralDiscreteScheduler,PNDMScheduler,EulerAncestralDiscreteScheduler,DPMSolverSinglestepScheduler,DDPMScheduler,DPMSolverMultistepScheduler,LMSDiscreteScheduler,DDIMScheduler,KDPM2DiscreteScheduler,HeunDiscreteScheduler,EulerDiscreteScheduler,
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)

from compel import Compel, ReturnedEmbeddingsType


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_CACHE = "diffusers-cache"
SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_MODEL_ID,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        )
        self.base = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            safety_checker=safety_checker,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
            use_safetensors=True
        ).to("cuda")
    
    
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to("cuda")    
        

        self.compel_proc  = Compel(tokenizer=[self.base.tokenizer, self.base.tokenizer_2] , text_encoder=[self.base.text_encoder, self.base.text_encoder_2], returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED, requires_pooled=[False, True])

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="a photo of an astronaut riding a horse on mars",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default='bad, ugly',
        ),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576,
                     640, 704, 768, 832, 896, 960, 1024],
            default=768,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576,
                     640, 704, 768, 832, 896, 960, 1024],
            default=768,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=8,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=21
        ),
        high_noise_frac: float = Input(default=0.4, le=1),
        
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        scheduler: str = Input(
            default="DPMSolverMultistep",
            choices=[
                "PNDM",
                "KLMS",
                "DDIM",
                "K_EULER",
                "DDPMScheduler",
                "K_EULER_ANCESTRAL",
                "DPMSolverMultistep",
                "DPMSolverSDEScheduler",
                "DEISMultistepScheduler",
                "UniPCMultistepScheduler",
                "KDPM2DiscreteScheduler",
                "HeunDiscreteScheduler",
                "DPMSolverSinglestepScheduler",
                "KDPM2AncestralDiscreteScheduler"
            ],
            description="Choose a scheduler.",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        # if width * height > 786432:
        #     raise ValueError(
        #         "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
        #     )

        self.pipe.scheduler = make_scheduler(
            scheduler, self.pipe.scheduler.config)

        prompt_embeds, pooled_prompt_embeds = self.compel_proc(prompt)
        negative_prompt_embeds = self.compel_proc(negative_prompt)
        generator = torch.Generator("cuda").manual_seed(seed)
        
        
        image = self.base(
            width=width,
            height=height,
            output_type="latent",
            generator=generator,
            guidance_scale=guidance_scale,
            denoising_end=high_noise_frac,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_outputs
        ).images


        output = self.refiner(
            image=image,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            denoising_start=high_noise_frac,
            num_images_per_prompt=num_outputs
        )

        output_paths = []
        for i, sample in enumerate(output.images):
            if output.nsfw_content_detected and output.nsfw_content_detected[i]:
                continue

            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
        "DPMSolverSDEScheduler": DPMSolverSDEScheduler.from_config(config),
        "DEISMultistepScheduler": DEISMultistepScheduler.from_config(config),
        "UniPCMultistepScheduler": UniPCMultistepScheduler.from_config(config),
        "KDPM2AncestralDiscreteScheduler": KDPM2AncestralDiscreteScheduler.from_config(config),
        "DPMSolverSinglestepScheduler": DPMSolverSinglestepScheduler.from_config(config),
        "DDPMScheduler": DDPMScheduler.from_config(config),
        "KDPM2DiscreteScheduler": KDPM2DiscreteScheduler.from_config(config),
        "HeunDiscreteScheduler": HeunDiscreteScheduler.from_config(config),

    }[name]
