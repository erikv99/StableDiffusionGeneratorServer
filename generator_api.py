from flask_restful import Resource, reqparse
import json

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

        print(json.dumps(args, indent=4))

        return {"message": "Image generated", "details": args}