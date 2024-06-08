import io
import os
import datetime
from flask import request, send_file
from flask_restful import Resource, reqparse

from generator_settings import GeneratorSettings
from generator import Generator

class GeneratorAPI(Resource):

    SAVE_TO_SERVER = True

    def __init__(self):
        
        # Since I am testing on an NVIDIA RTX A4000, this code will only use a single generation.
        # Multi-generator support will be added later if deemed necessary.
        self._generator =  Generator()
        print("\nGenerator API initialized.\n")

    def get(self):
        
        # TODO: Maybe it is nice to have get request return a image with last set settings or default.
        return {"message": "Welcome to the generator API. Please use the POST method to generate images"}

    def post(self):

        print(request.json)  # This will log the incoming request data


        parser = reqparse.RequestParser()
        parser.add_argument('Prompt', type=str, required=True, help='Prompt is required')
        parser.add_argument('NegativePrompt', type=str, required=True, help='Negative prompt is required')
        parser.add_argument('InferenceSteps', type=int, required=True, help='Number of steps is required')
        parser.add_argument('GuidanceScale', type=float, required=True, help='Guidance scale is required')
        parser.add_argument('StyleStrength', type=float, required=True, help='Style strength is required')
        args = parser.parse_args()

        # TODO: Log the request details to file.
        print("User Agent:", request.user_agent.string)

        settings = GeneratorSettings(
            args.Prompt, 
            args.NegativePrompt, 
            args.GuidanceScale, 
            args.StyleStrength, 
            args.InferenceSteps)

        try:
            self._generator.set_settings(settings)

            # Add develop mode which returns default image.

            image = self._generator.generate_image()
        except Exception as e:
            return {"error": str(e)}
        
        if self.SAVE_TO_SERVER:
            self._save_img_to_server(image)

        # Log the generated image details
        print("Generated Image Details:")
        print("  - Size:", image.size)
        print("  - Mode:", image.mode)
        print("  - Format:", image.format)

        # TODO: This is for easier viewing of the image in postman, definitely not needed in production
        image = image.resize((512, 512))

        image.format = "PNG" # TODO: CHECK IF THIS IS NEEDED
        
        # Convert the image to bytes for sending it back
        bytes_io = io.BytesIO()
        image.save(bytes_io, 'PNG')
        bytes_io.seek(0)

        return send_file(bytes_io, mimetype='image/png')

    def _save_img_to_server(self, image):

        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        output_path = os.path.join(output_dir, f"generated_image_{timestamp}.png")
        image.save(output_path)