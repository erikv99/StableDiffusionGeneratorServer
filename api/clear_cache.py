from flask_restful import Resource
import torch
import gc

class ClearCache(Resource):

    def post(self):

        print("\nClear cache request received.\n")
        torch.cuda.empty_cache()
        gc.collect()
        return {"message": "Cache cleared."}, 200