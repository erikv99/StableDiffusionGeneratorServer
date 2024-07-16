from flask_restful import Resource, reqparse

from generator_settings import GeneratorSettings
from queue_handler import QueueHandler

class EnqueueGeneration(Resource):

    # SAVE_TO_SERVER = True

    def __init__(self, queue_handler: QueueHandler):
        
        self._queue_handler = queue_handler
        print("\nGenerator API initialized.\n")

    def post(self):

        parser = reqparse.RequestParser()
        parser.add_argument('Prompt', type=str, required=True, help='Prompt is required')
        parser.add_argument('NegativePrompt', type=str, required=True, help='Negative prompt is required')
        parser.add_argument('InferenceSteps', type=int, required=True, help='Number of steps is required')
        parser.add_argument('GuidanceScale', type=float, required=True, help='Guidance scale is required')
        parser.add_argument('StyleStrength', type=float, required=True, help='Style strength is required')
        args = parser.parse_args()

        settings = GeneratorSettings(
            args.Prompt, 
            args.NegativePrompt, 
            args.GuidanceScale, 
            args.StyleStrength, 
            args.InferenceSteps)

        try:
            request_uuid = self._queue_handler.enqueue_request(settings)
            return {"uuid": request_uuid}, 202

        except Exception as e:
            return {"error": str(e.args[0])}, 400