from PIL import Image
from open_webui.backend_customer.tryon.src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from open_webui.backend_customer.tryon.src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from open_webui.backend_customer.tryon.src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler,AutoencoderKL
from typing import List

import torch
import os, io, zipfile, base64
from transformers import AutoTokenizer
import numpy as np
from open_webui.backend_customer.tryon.utils_mask import get_mask_location
from torchvision import transforms
import open_webui.backend_customer.tryon.apply_net
from open_webui.backend_customer.tryon.preprocess.humanparsing.run_parsing import Parsing
from open_webui.backend_customer.tryon.preprocess.openpose.run_openpose import OpenPose
from open_webui.backend_customer.tryon.detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
from ..plm_enum import RequestKey, ResponseKey
from .apply_net import create_argument_parser
class PLMTryon:
    def __init__(self) -> None:
        base_path = 'yisol/IDM-VTON'

        unet = UNet2DConditionModel.from_pretrained(
            base_path,
            subfolder="unet",
            torch_dtype=torch.float16,
        )
        tokenizer_one = AutoTokenizer.from_pretrained(
            base_path,
            subfolder="tokenizer",
            revision=None,
            use_fast=False,
        )
        tokenizer_two = AutoTokenizer.from_pretrained(
            base_path,
            subfolder="tokenizer_2",
            revision=None,
            use_fast=False,
        )
        noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

        text_encoder_one = CLIPTextModel.from_pretrained(
            base_path,
            subfolder="text_encoder",
            torch_dtype=torch.float16,
        )
        text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
            base_path,
            subfolder="text_encoder_2",
            torch_dtype=torch.float16,
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_path,
            subfolder="image_encoder",
            torch_dtype=torch.float16,
            )
        vae = AutoencoderKL.from_pretrained(base_path,
                                            subfolder="vae",
                                            torch_dtype=torch.float16,
        )

        # "stabilityai/stable-diffusion-xl-base-1.0",
        UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
            base_path,
            subfolder="unet_encoder",
            torch_dtype=torch.float16,
        )

        self._parsing_model = Parsing(0)
        self._openpose_model = OpenPose(0)

        UNet_Encoder.requires_grad_(False)
        image_encoder.requires_grad_(False)
        vae.requires_grad_(False)
        unet.requires_grad_(False)
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        self._tensor_transfrom = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ]
            )

        self._pipe = TryonPipeline.from_pretrained(
                base_path,
                unet=unet,
                vae=vae,
                feature_extractor= CLIPImageProcessor(),
                text_encoder = text_encoder_one,
                text_encoder_2 = text_encoder_two,
                tokenizer = tokenizer_one,
                tokenizer_2 = tokenizer_two,
                scheduler = noise_scheduler,
                image_encoder=image_encoder,
                torch_dtype=torch.float16,
        )
        self._pipe.unet_encoder = UNet_Encoder

    def _pil_to_binary_mask(self, pil_image, threshold=0):
        np_image = np.array(pil_image)
        grayscale_image = Image.fromarray(np_image).convert("L")
        binary_mask = np.array(grayscale_image) > threshold
        mask = np.zeros(binary_mask.shape, dtype=np.uint8)
        for i in range(binary_mask.shape[0]):
            for j in range(binary_mask.shape[1]):
                if binary_mask[i,j] == True :
                    mask[i,j] = 1
        mask = (mask*255).astype(np.uint8)
        output_mask = Image.fromarray(mask)
        return output_mask

    def _tryon(self, human_img,garm_img,garment_des,is_checked,is_checked_crop,denoise_steps,seed, garm_category):
        device = "cpu" # cuda for GPU 
        
        self._openpose_model.preprocessor.body_estimation.model.to(device)
        # uncomment followinfg 2 line for GPU
        # self._pipe.to(device)
        # self._pipe.unet_encoder.to(device)

        garm_img= garm_img.convert("RGB").resize((768,1024))
        human_img_orig = human_img.convert("RGB")    
        
        if is_checked_crop:
            width, height = human_img_orig.size
            target_width = int(min(width, height * (3 / 4)))
            target_height = int(min(height, width * (4 / 3)))
            left = (width - target_width) / 2
            top = (height - target_height) / 2
            right = (width + target_width) / 2
            bottom = (height + target_height) / 2
            cropped_img = human_img_orig.crop((left, top, right, bottom))
            crop_size = cropped_img.size
            human_img = cropped_img.resize((768,1024))
        else:
            human_img = human_img_orig.resize((768,1024))


        if is_checked:
            keypoints = self._openpose_model(human_img.resize((384,512)))
            model_parse, _ = self._parsing_model(human_img.resize((384,512)))
            mask, mask_gray = get_mask_location('hd', garm_category, model_parse, keypoints)
            mask = mask.resize((768,1024))
        #else:
            #mask = self._pil_to_binary_mask(dict['layers'][0].convert("RGB").resize((768, 1024)))
            # mask = transforms.ToTensor()(mask)
            # mask = mask.unsqueeze(0)
        mask_gray = (1-transforms.ToTensor()(mask)) * self._tensor_transfrom(human_img)
        mask_gray = to_pil_image((mask_gray+1.0)/2.0)


        human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
        human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
        
        

        args = create_argument_parser().parse_args((
            'show', 
            'open_webui/backend_customer/tryon/configs/densepose_rcnn_R_50_FPN_s1x.yaml', 
            'open_webui/backend_customer/tryon/ckpt/densepose/model_final_162be9.pkl', 
            'dp_segm', 
            '-v', 
            '--opts', 'MODEL.DEVICE', device)) 
        # verbosity = getattr(args, "verbosity", None)
        pose_img = args.func(args,human_img_arg)    
        pose_img = pose_img[:,:,::-1]    
        pose_img = Image.fromarray(pose_img).resize((768,1024))
        
        # un comment folowings for GPU
        # with torch.no_grad():
        #     # Extract the images
        #     with torch.amp.autocast("cuda"):
        #         with torch.no_grad():
        #             prompt = "model is wearing " + garment_des
        #             negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        #             with torch.inference_mode():
        #                 (
        #                     prompt_embeds,
        #                     negative_prompt_embeds,
        #                     pooled_prompt_embeds,
        #                     negative_pooled_prompt_embeds,
        #                 ) = self._pipe.encode_prompt(
        #                     prompt,
        #                     num_images_per_prompt=1,
        #                     do_classifier_free_guidance=True,
        #                     negative_prompt=negative_prompt,
        #                 )
                                        
        #                 prompt = "a photo of " + garment_des
        #                 negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        #                 if not isinstance(prompt, List):
        #                     prompt = [prompt] * 1
        #                 if not isinstance(negative_prompt, List):
        #                     negative_prompt = [negative_prompt] * 1
        #                 with torch.inference_mode():
        #                     (
        #                         prompt_embeds_c,
        #                         _,
        #                         _,
        #                         _,
        #                     ) = self._pipe.encode_prompt(
        #                         prompt,
        #                         num_images_per_prompt=1,
        #                         do_classifier_free_guidance=False,
        #                         negative_prompt=negative_prompt,
        #                     )



        #                 pose_img =  self._tensor_transfrom(pose_img).unsqueeze(0).to(device,torch.float16)
        #                 garm_tensor =  self._tensor_transfrom(garm_img).unsqueeze(0).to(device,torch.float16)
        #                 generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
        #                 images = self._pipe(
        #                     prompt_embeds=prompt_embeds.to(device,torch.float16),
        #                     negative_prompt_embeds=negative_prompt_embeds.to(device,torch.float16),
        #                     pooled_prompt_embeds=pooled_prompt_embeds.to(device,torch.float16),
        #                     negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device,torch.float16),
        #                     num_inference_steps=denoise_steps,
        #                     generator=generator,
        #                     strength = 1.0,
        #                     pose_img = pose_img.to(device,torch.float16),
        #                     text_embeds_cloth=prompt_embeds_c.to(device,torch.float16),
        #                     cloth = garm_tensor.to(device,torch.float16),
        #                     mask_image=mask,
        #                     image=human_img, 
        #                     height=1024,
        #                     width=768,
        #                     ip_adapter_image = garm_img.resize((768,1024)),
        #                     guidance_scale=2.0,
        #                 )[0]

        # if is_checked_crop:
        #     out_img = images[0].resize(crop_size)        
        #     human_img_orig.paste(out_img, (int(left), int(top)))    
        #     return human_img_orig, mask_gray
        # else:
        #     return images[0], mask_gray
        return human_img_orig, mask_gray

    def do(self, request_paras):
        garment_category ="upper_body"
        if RequestKey.person_64string.name in request_paras:
            person_64string = request_paras[RequestKey.person_64string.name]
            person_bytes = base64.b64decode(person_64string)
            person_image = Image.open(io.BytesIO(person_bytes))
        if RequestKey.garment_64string.name in request_paras:
            garment_64string = request_paras[RequestKey.garment_64string.name]
            garment_bytes = base64.b64decode(garment_64string)
            garment_image = Image.open(io.BytesIO(garment_bytes))
        if RequestKey.garment_category.name in request_paras:
            garment_category = request_paras[RequestKey.garment_category.name]
        
        
        garment_new, mask_gray = self._tryon(
            human_img=person_image,
            garm_img= garment_image,
            garment_des="",
            is_checked=True,
            is_checked_crop=False,
            denoise_steps=0,
            seed=42,
            garm_category = garment_category
        )
        #  convert image to 64 string
        buffered = io.BytesIO()
        mask_gray.save(buffered, format="JPEG")
        # un comment following for GPU
        # garment_new.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        base64_bytes = base64.b64encode(image_bytes)
        base64_string = base64_bytes.decode('utf-8')
        
        return {ResponseKey.image_64string.name: base64_string}

