import os
import torch
import torchvision

from huggingface_hub import hf_hub_download
from PhotoMaker.photomaker.pipeline import PhotoMakerStableDiffusionXLPipeline
from generatorSettings import GeneratorSettings
from PIL import Image
from diffusers.utils import load_image
from diffusers import DDIMScheduler
import random

# TODO: add logging.

# Not sure if i like the Generator as a object approach. 
# TODO: Consider a function based approach.

class Generator:

    # TODO: make base_model_path changeable by endpoint so it can be change client side.
    # base_model_path = 'SG161222/RealVisXL_V4.0'
    # base_model_path = "SG161222/RealVisXL_V4.0_Lightning"
    BASE_MODEL_PATH = "RunDiffusion/Juggernaut-XL-v9"
    DEVICE = "cuda"
    DEFAULT_IMAGE_DIR = "./default_images"

    @staticmethod
    def retrieve_cuda_info():

        cuda_version = torch.version.cuda
        cuda_id = torch.cuda.current_device()
        cuda_device_name = torch.cuda.get_device_name(cuda_id)
        torch_version = torch.__version__
        return cuda_version, cuda_id, cuda_device_name, torch_version

    @staticmethod
    def empty_cache():
        torch.cuda.empty_cache()

    # TODO: Consider moving to download helper/service
    def _retrieve_photomaker(self) -> str:
        """
        Retrieves the cached PhotoMaker model checkpoint from the TencentARC/PhotoMaker repository.
        If not cached, downloads the latest version from the repository.

        Returns:
            str: Local path of file or if networking is off, last version of file cached on disk.
        """

        return hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")

    # TODO: Clean up, split in to logical (easier digestable) parts and add comments.
    def _setup_pipeline(self):

        pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            self.BASE_MODEL_PATH,
            torch_dtype=torch.float16, # TODO expirement with different torch dtypes
            use_safetensors=True,
            variant="fp16"
        )

        pipe.to(self.DEVICE)

        photomaker_model_path = self._retrieve_photomaker()
        weight_name = os.path.basename(photomaker_model_path)

        pipe.load_photomaker_adapter(
            pretrained_model_name_or_path_or_dict=photomaker_model_path,
            weight_name=weight_name
        )

        pipe.id_encoder.to(self.DEVICE)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.fuse_lora()

    # TODO: Move to image loading stuff to it's own module
    def _load_default_images(self):

        default_images = []

        for filename in os.listdir(self.DEFAULT_IMAGE_DIR):
            image_path = os.path.join(self.DEFAULT_IMAGE_DIR, filename)
            image = load_image(image_path)
            default_images.append(image)

        return default_images

    def _get_input_dirs(self):
        """
        Get a list of all non default input directories. 

        TODO: Add validation checks on each dir to see if they are valid for usage depending on the requirements for usage.

        Returns:
            A list of all non default input directories.
        """

        def _is_valid(folder):
            return os.path.isdir(os.path.join(self.DEFAULT_IMAGE_DIR, folder)) and folder != 'default'
        
        valid_input_dirs = [_is_valid(folder) for folder in os.listdir(self.DEFAULT_IMAGE_DIR)]
        return valid_input_dirs if any(valid_input_dirs) else []


    def _load_input_images(self):
        """
        Loads input images from the input directories in to self.input_images.
        """

        input_dirs = self._get_input_dirs()

        if not input_dirs:
            return self._load_default_images()

        # TODO: Add input through the API (give image base64's as input? upload endpoint?)
        # TODO: Available input folder selection of by API

        # NOTE: For now, we will randomly select a folder from the available input directories.
        random_dir = random.choice(input_dirs)
        path = os.path.join(self.DEFAULT_IMAGE_DIR, random_dir)
        input_images = []
        
        for filename in os.listdir(path):
        
            image_path = os.path.join(path, filename)
            image = load_image(image_path)
            input_images.append(image)
        
        self.input_images = input_images

    def __init__(self):
        
        # TODO: add gpu only check.

        self._print_cuda_info()
        self._setup_pipeline()
        self._load_input_images()

        # TODO

    def _print_cuda_info(self):
        cuda_version, cuda_id, cuda_device_name, torch_version = self._retrieve_cuda_info()
        print(f'CUDA version: {cuda_version}')
        print(f"ID of current CUDA device: {cuda_id}")
        print(f"Name of current CUDA device: {cuda_device_name}")
        print(f'Torch version: {torch_version}')

    def generate_image(self, settings: GeneratorSettings):

        pass
