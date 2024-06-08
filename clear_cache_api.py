from flask_restful import Resource
import torch

class ClearCacheAPI(Resource):

    def post(self):

        print("\nClear cache request received.\n")
        torch.cuda.empty_cache()
        return {"message": "Cache cleared."}, 200