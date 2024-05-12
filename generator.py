from generatorSettings import GeneratorSettings
import torch
import torchvision

# TODO: add logging.

class Generator:

    def _retrieve_cuda_info():

        cuda_version = torch.version.cuda
        cuda_id = torch.cuda.current_device()
        cuda_device_name = torch.cuda.get_device_name(cuda_id)
        torch_version = torch.__version__
        return cuda_version, cuda_id, cuda_device_name, torch_version

    def _empty_cache():
        torch.cuda.empty_cache()

    def __init__(self):
        
        # Todo: add gpu only check.

        cuda_version, cuda_id, cuda_device_name, torch_version = self._retrieve_cuda_info()
        print(f'CUDA version: {cuda_version}')
        print(f"ID of current CUDA device: {cuda_id}")
        print(f"Name of current CUDA device: {cuda_device_name}")
        print(f'Torch version: {torch_version}')
        pass

    def generate_image(self, settings: GeneratorSettings):


        pass
