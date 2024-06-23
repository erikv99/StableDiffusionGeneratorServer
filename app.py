import datetime
import time
from flask import Flask
from flask_restful import Api

from api.clear_cache import ClearCache
from api.enqueue_generation import EnqueueGeneration
from api.retrieve_output import RetrieveOutput
from generator import Generator

import os

from generator_settings import GeneratorSettings

app = Flask(__name__)
api = Api(app)

# This is temporary, input will be handled differently later.
def _validate_input_dir():

    if not os.path.exists("./input"):

        print("Input directory not found. Creating input directory.")
        os.makedirs("./input")

    if not os.path.exists("./input/default"):

        print("Default input directory not found. Creating default input directory.")
        os.makedirs("./input/default")

    # Todo: Add some default images to the default input directory i guess

def _validate_output_dir():

    if not os.path.exists("./output"):

        print("Output directory not found. Creating output directory.")
        os.makedirs("./output")

def main():
    
    _validate_output_dir()
    _validate_input_dir()
    
    # generator = Generator()       
      
    # settings = GeneratorSettings(
    #     Prompt="A young happy male img standing on the edge of a cliff, facting the camera",
    #     NegativePrompt=" missing fingers, (blurred background, unsharp background, render, drawing, anime, bad photo, bad photography:1.3), (worst quality, low quality, blurry:1.2), (bad teeth, deformed teeth, deformed lips), (bad anatomy, bad proportions:1.1), (deformed iris, deformed pupils), (deformed eyes, bad eyes), (deformed face, bad face), (ugly facial hair, goody haircut, inconsistent hairstyle) (deformed hands, bad hands, fused fingers), morbid, mutilated, mutation, disfigured", 
    #     InferenceSteps= 22,
    #     GuidanceScale= 2.5,
    #     StyleStrength= 1.3
    # )

    # image = generator._process_request(settings)
    
    # def _save_img_to_server(image):

    #     output_dir = "./output"
    #     os.makedirs(output_dir, exist_ok=True)
    #     timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    #     output_path = os.path.join(output_dir, f"generated_image_{timestamp}.png")
    #     image.save(output_path)
        
    # _save_img_to_server(image)
    
    # while not generator.is_ready():
    #     print("Waiting for generator to be ready.")
    #     time.sleep(5)
    
    generator = Generator()
    
    api.add_resource(EnqueueGeneration, '/enqueue-generation', resource_class_kwargs={'generator': generator})
    api.add_resource(RetrieveOutput, '/retrieve-output/<string:id>', resource_class_kwargs={'generator': generator})
    api.add_resource(ClearCache, '/clear-cache')
    # api.ap
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)

if __name__ == "__main__":
    main()
