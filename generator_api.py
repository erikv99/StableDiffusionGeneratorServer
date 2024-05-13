from flask_restful import Resource, reqparse
import json

from genarator_settings import GeneratorSettings
from generator import Generator
from flask import request, send_file

class GeneratorAPI(Resource):
    
    def get(self):
        return {"message": "Welcome to the generator API. Please use the POST method to generate images"}

    def post(self):

        parser = reqparse.RequestParser()
        parser.add_argument('prompt', type=str, required=True, help='Prompt is required')
        parser.add_argument('negative_prompt', type=str, required=True, help='Negative prompt is required')
        parser.add_argument('steps', type=int, required=True, help='Number of steps is required')
        parser.add_argument('guidance_scale', type=float, required=True, help='Guidance scale is required')
        parser.add_argument('style_strength', type=float, required=True, help='Style strength is required')
        args = parser.parse_args()

        # TODO: Add actual logging
        print("IP Address:", request.remote_addr)
        print("User Agent:", request.user_agent.string)

        settings = GeneratorSettings(
            args.prompt, 
            args.negative_prompt, 
            args.steps, 
            args.guidance_scale, 
            args.style_strength)

        # Since I am testing on an NVIDIA RTX A4000, this code will only use a single generation.
        # Multi-generator support will be added later if deemed necessary.
        generator = Generator()

        image = generator.generate_image(settings)

        # TODO: Log creation details, intended for checking any problems / errors

        return send_file(image, mimetype='image/png')