import os
import torch

from huggingface_hub import hf_hub_download
from PhotoMaker.photomaker import PhotoMakerStableDiffusionXLPipeline
from diffusers.utils import load_image
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, DPMSolverSDEScheduler
import random

from generator_settings import GeneratorSettings

# TODO: add logging.

# Not sure if i like the Generator as a object approach. 
# TODO: Consider a function based approach.

class Generator:

    # TODO: make base_model_path changeable by endpoint so it can be change client side.
    
    BASE_MODEL_PATH = "SG161222/RealVisXL_V4.0_Lightning"
 
    DEVICE = "cuda"
    DEFAULT_IMAGE_DIR = "./input/default"
    INPUT_DIR = "./input"

    def __init__(self):
        
        print(torch.cuda.is_available())
        print(torch.cuda.current_device())
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
        self.pipe = None
        self.input_images = []

        # TODO: add gpu only check.
        self.empty_cache()
        self._print_cuda_info()
        self._setup_pipeline()
        self._load_input_images()

        print("Generator initialized.")

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

        try:
            self.device = torch.device(self.DEVICE)
            print(f"Using device: {self.device}")

            self.pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
                self.BASE_MODEL_PATH,
                torch_dtype=torch.float16,  # TODO: experiment with different torch dtypes
                use_safetensors=True,
                variant="fp16"
            )

            self.pipe.to(self.device)

            photomaker_model_path = self._retrieve_photomaker()
            weight_name = os.path.basename(photomaker_model_path)

            self.pipe.load_photomaker_adapter(
                pretrained_model_name_or_path_or_dict=photomaker_model_path,
                weight_name=weight_name
            )

            if hasattr(self.pipe, 'id_encoder'):
                
                self.pipe.id_encoder.to(self.device)

            else:
                
                raise AttributeError("Pipeline does not have an attribute 'id_encoder'")

            # DPMSolverMultistepScheduler
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.fuse_lora()
            
            print("Pipeline setup completed successfully.")
        
        except Exception as e:
            print(f"Error during pipeline setup: {e}")
            raise

    def set_settings(self, settings: GeneratorSettings):

        self._settings = settings

    def generate_image(self):

        if self._settings is None:
            raise ValueError("No settings have been set for the generator. Please set settings before generating an image.")

        generator = torch.Generator(device=self.device).manual_seed(torch.randint(0, 1000000, (1,)).item())

        start_merge_step = int(float(self._settings.style_strength) / 100 * self._settings.number_of_steps)
        if start_merge_step > 30:
            start_merge_step = 30

        images = self.pipe(
            prompt=self._settings.prompt,
            input_id_images=self.input_images,
            negative_prompt=self._settings.negative_prompt,
            num_images_per_prompt=1,
            num_inference_steps=self._settings.number_of_steps,
            start_merge_step=start_merge_step,
            generator=generator,
            guidance_scale=self._settings.guidance_scale,
            height=512,
            width=512
        ).images

        # NOTE: This is temporary and should be removed if ever adding multiple image generation support.
        return images[0]

    # TODO: Move to image loading stuff to it's own module
    def _load_default_images(self):

        default_images = []

        for filename in os.listdir(self.DEFAULT_IMAGE_DIR):
            image_path = os.path.join(self.DEFAULT_IMAGE_DIR, filename)
            image = load_image(image_path)
            default_images.append(image)

        if (len(default_images) == 0):
            raise ValueError("No default images found in the default image directory.")

        return default_images

    def _get_input_dirs(self):
        """
        Get a list of all non default input directories. 

        TODO: Add validation checks on each dir to see if they are valid for usage depending on the requirements for usage.

        Returns:
            A list of all non default input directories.
        """

        def _is_valid(folder):
            return os.path.isdir(os.path.join(self.INPUT_DIR, folder)) and folder != 'default'
        
        dirs = os.listdir(self.INPUT_DIR)
        valid_input_dirs = [folder for folder in dirs if _is_valid(folder)]
        return valid_input_dirs if any(valid_input_dirs) else []

    def _load_input_images(self):
        """
        Loads input images from the input directories in to self.input_images.
        """

        input_dirs = self._get_input_dirs()

        # TODO: This crashes when no input directories are available since input/default and its content is not added to repo
        if not input_dirs:
            self.input_images = self._load_default_images()
            return

        # TODO: Add input through the API (give image base64's as input? upload endpoint?)
        # TODO: Available input folder selection of by API

        # NOTE: For now, we will randomly select a folder from the available input directories.
        random_dir = random.choice(input_dirs)
        path = os.path.join(self.INPUT_DIR, random_dir)
        input_images = []
        
        for filename in os.listdir(path):
        
            image_path = os.path.join(path, filename)
            image = load_image(image_path)
            input_images.append(image)
        
        self.input_images = input_images
        print(f"Number of input images: {len(input_images)}")

    def _print_cuda_info(self):
        cuda_version, cuda_id, cuda_device_name, torch_version = self.retrieve_cuda_info()
        print(f'CUDA version: {cuda_version}')
        print(f"ID of current CUDA device: {cuda_id}")
        print(f"Name of current CUDA device: {cuda_device_name}")
        print(f'Torch version: {torch_version}')