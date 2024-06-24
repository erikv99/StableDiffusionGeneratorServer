import os
import time
import uuid
import torch
import queue
import threading
from huggingface_hub import hf_hub_download
from PhotoMaker.photomaker import PhotoMakerStableDiffusionXLPipeline
from diffusers.utils import load_image
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, DPMSolverSDEScheduler
import random
from enum import Enum
from generator_settings import GeneratorSettings

class Generator:
    class GeneratorStatus(Enum):
        Off = 0
        Initializing = 1
        Error = 2
        Available = 3
        Generating = 4

    BASE_MODEL_PATH = "SG161222/RealVisXL_V4.0_Lightning"
    DEVICE = "cuda"
    DEFAULT_IMAGE_DIR = "./input/default"
    INPUT_DIR = "./input"
    QUEUE_CHECK_INTERVAL_SECONDS = 20
    DEBUG = True
    
    def __init__(self):
        
        print("Initializing generator...")
        
        self._pipe = None
        self._input_images = []
        self._results = {}
        self._status = self.GeneratorStatus.Initializing
        self._request_queue: queue[str, GeneratorSettings] = queue.Queue()

        print("Clearing Cuda cache...")
        # torch.cuda.empty_cache()

        self._setup_pipeline()

        if not self.DEBUG and self.DEVICE == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available on this device.")
        
        # print("Loading input images...")
        # self._load_input_images()

        print("Generator initialized.")
        self._status = self.GeneratorStatus.Available
        self._thread = threading.Thread(target=self._process_queue, daemon = True)
        self._thread.start()

    @staticmethod
    def empty_cuda_cache_if_threshold_reached(threshold_ratio=0.2): 
        """
            Empties the CUDA cache if the free memory is below a certain threshold.
            Threshold is calculated as a ratio of the total memory available on the device.
            E.g. if the threshold ratio is 0.2, the threshold will be 20% of the total memory available on the device.
        """
    
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        allocated_memory = torch.cuda.memory_allocated(0)
        
        free_memory = reserved_memory - allocated_memory
        threshold = total_memory * threshold_ratio

        if free_memory < threshold:

            print("Memory below threshold. Clearing CUDA cache...")
            # torch.cuda.empty_cache()

    @staticmethod
    def retrieve_cuda_info():
        cuda_version = torch.version.cuda
        cuda_id = torch.cuda.current_device()
        cuda_device_name = torch.cuda.get_device_name(cuda_id)
        torch_version = torch.__version__

        return cuda_version, cuda_id, cuda_device_name, torch_version

    def is_ready(self):
        return self._status is self.GeneratorStatus.Available

    # TODO: Consider moving to download helper/service
    def _retrieve_photomaker(self) -> str:
        """
        Retrieves the cached PhotoMaker model checkpoint from the TencentARC/PhotoMaker repository.
        If not cached, downloads the latest version from the repository.

        Returns:
            str: Local path of file or if networking is off, last version of file cached on disk.
        """

        return hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")

    def _process_queue(self):
        
        while True:
            
            while self._status in [self.GeneratorStatus.Initializing, self.GeneratorStatus.Generating]:
                print(f"Generator is {self._status.name}. Waiting...")
                time.sleep(10)

            try:
                
                request_uuid, data = self._request_queue.get(timeout=self.QUEUE_CHECK_INTERVAL_SECONDS)
                print(f"Processing request #{request_uuid}")

                retries = 0
                max_retries = 3
                result = None

                while retries < max_retries:
                    try:
                        result = self._process_request(data)
                        self._results[request_uuid] = result
                        break
                    except torch.cuda.OutOfMemoryError as e:
                        retries += 1
                        print(f"Out of memory error: {e}. Retrying ({retries}/{max_retries}) after clearing cache...")
                        torch.cuda.empty_cache()
                        time.sleep(10)

                if retries == max_retries:
                    print(f"Retry failed after {max_retries} attempts")
                    self._results[request_uuid] = "Out of memory error"

                self._request_queue.task_done()
            except queue.Empty:
                print("Queue is empty. Continuing to wait...")

            if not self.DEBUG:
                torch.cuda.memory_summary(device=None, abbreviated=False)

    def get_result(self, request_uuid):
        
        if request_uuid in self._results: 
            return self._results.get(request_uuid) 
        
        raise ValueError(f"Request with UUID {request_uuid} not found.")

    def _process_request(self, settings):
        
        self._status = self.GeneratorStatus.Generating
        
        try:
            if self.DEBUG:
                print(f"Mock Processing request: {settings}")
                return None
            else:
                return self.generate_image(settings)
        finally:
            self._status = self.GeneratorStatus.Available
    
    def clear_queue(self):
        self._request_queue.queue.clear()

    def _load_base_model(self):
        
        self._pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0_Lightning",
            torch_dtype=torch.float16,  
            use_safetensors=True,
            variant="fp16"
        )

    def _load_photomaker_adapter(self):
        
        photomaker_model_path = self._retrieve_photomaker()
        weight_name = os.path.basename(photomaker_model_path)
        
        self._pipe.load_photomaker_adapter(
            pretrained_model_name_or_path_or_dict=photomaker_model_path,
            weight_name=weight_name
        )
        
        if hasattr(self._pipe, 'id_encoder'):
            self._pipe.id_encoder.to(self.device)
        else:
            raise AttributeError("Pipeline does not have an attribute 'id_encoder'")

    def _initialize_device(self):
        self.device = torch.device(self.DEVICE)
        print(f"Using device: {self.device}")

    # TODO: Clean up, split in to logical (easier digestable) parts and add comments.
    def _setup_pipeline(self):

        if self.DEBUG:
            print("Running in debug mode. Skipping pipeline setup.")
            return

        print("Setting up pipeline...")
        
        try:
            
            self._initialize_device()
            self._load_base_model()
            self._load_photomaker_adapter()

            self._pipe.to(self.device)

            # photomaker_model_path = self._retrieve_photomaker()
            # weight_name = os.path.basename(photomaker_model_path)

            # self._pipe.load_photomaker_adapter(
            #     pretrained_model_name_or_path_or_dict=photomaker_model_path,
            #     weight_name=weight_name
            # )

            # if hasattr(self._pipe, 'id_encoder'):
            #     self._pipe.id_encoder.to(self.device)

            # else:
            #     raise AttributeError("Pipeline does not have an attribute 'id_encoder'")

            self._configure_scheduler()
            print("Pipeline setup completed successfully.")
        
        except Exception as e:
            print(f"Error during pipeline setup: {e}")
            raise
    
    def _configure_scheduler(self):
        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(self._pipe.scheduler.config)
        self._pipe.fuse_lora()
        
    def generate_image(self, settings: GeneratorSettings):
        if settings is None:
            raise ValueError("No settings have been provided.")

        generator = torch.Generator(device=self.DEVICE).manual_seed(torch.randint(0, 1000000, (1,)).item())

        # start_merge_step = int(float(settings.style_strength) / 100 * settings.number_of_steps)
        # if start_merge_step > 30:
        #     start_merge_step = 30
        start_merge_step = min(int(float(settings.style_strength) / 100 * settings.number_of_steps), 30)

        images = self._pipe(
            prompt=settings.prompt,
            input_id_images=self._input_images,
            negative_prompt=settings.negative_prompt,
            num_images_per_prompt=1,
            num_inference_steps=settings.number_of_steps,
            start_merge_step=start_merge_step,
            generator=generator,
            guidance_scale=settings.guidance_scale,
            height=512,
            width=512
        ).images
        return images[0]

    def get_result(self, request_uuid: str):
        return self._results.get(request_uuid)
    
    def enqueue_request(self, settings: GeneratorSettings) -> str:
        request_uuid = str(uuid.uuid4())
        self._request_queue.put((request_uuid, settings))
        return request_uuid
    
    # TODO: Move to image loading stuff to it's own module
    def _load_default_images(self):

        default_images = []

        for filename in os.listdir(self.DEFAULT_IMAGE_DIR):
            image_path = os.path.join(self.DEFAULT_IMAGE_DIR, filename)
            image = load_image(image_path)
            default_images.append(image)

        if (len(default_images) == 0):
            raise ValueError("No default images found in the default image directory. (add some images to ./input/default)")

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
            self._input_images = self._load_default_images()
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
        
        self._input_images = input_images
        print(f"Number of input images: {len(input_images)}")

    def _print_cuda_info(self):
        cuda_version, cuda_id, cuda_device_name, torch_version = self.retrieve_cuda_info()
        print(f'CUDA version: {cuda_version}')
        print(f"ID of current CUDA device: {cuda_id}")
        print(f"Name of current CUDA device: {cuda_device_name}")
        print(f'Torch version: {torch_version}')
