import torch
import os
import zipfile
from io import BytesIO
from torchvision.utils import save_image
from ts.torch_handler.base_handler import BaseHandler

ZIPFILE = "model_and_pretrained.zip"


class ModelHandler(BaseHandler):
    def __init__(self):
        self.initialized = False
        self.device = None
        self.store_avg = True
        self.model = None
        self.default_number_of_images = 1
        self.top_k = 128
        self.grid_size = 1

    def initialize(self, context):
        """
        Extract the models zip; Take the serialized file and load the model
        """
        print("initilization")
        if context is not None:
            properties = context.system_properties
            model_dir = properties.get("model_dir")
            gpu_id = properties.get("gpu_id")
        else:
            model_dir = "."

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # If not already extracted, Extract model source code and pretrained checkpoint
        if not os.path.exists(model_dir + "/models"):
            with zipfile.ZipFile(model_dir + "/" + ZIPFILE, "r") as zip_ref:
                zip_ref.extractall(model_dir)

        # Load Model
        from models.dalle.dalle import MinDalle
        print("import dalle")

        self.model = MinDalle(
            is_mega=True,  # TODO: need udpate
            root_dir=model_dir + "/pretrained",
            device=self.device,
            dtype=torch.float16,
        )
        # state_dict = torch.load(model_dir + "/" + CHECKPOINT, map_location=self.map_location)
        # self.dcgan_model.load_state_dict(state_dict)

        self.initialized = True

    def preprocess(self, requests):
        """
        Build noise data by using "number of images" and other "constraints" provided by the end user.
        """
        print("preprocess")
        preprocessed_data = []
        for req in requests:
            data = (
                req.get("data") if req.get("data") is not None else req.get("body", {})
            )

            input_text = data.get(
                "input_text", ""
            )  # TODO: could replace with better default input text

            preprocessed_data.append({"input": input_text})
        return preprocessed_data

    def inference(self, preprocessed_data, *args, **kwargs):
        """
        Take the noise data as an input tensor, pass it to the model and collect the output tensor.
        """
        print("inference")
        # input_batch = torch.cat(tuple(map(lambda d: d["input"], preprocessed_data)), 0)
        # with torch.no_grad():
        #     image_tensor = self.dcgan_model.test(input_batch, getAvG=True, toCPU=True)
        # output_batch = torch.split(image_tensor, tuple(map(lambda d: d["number_of_images"], preprocessed_data)))
        input_text = list(map(lambda d: d["input"], preprocessed_data))[-1]
        output_img = self.model.generate_image(input_text, self.grid_size, top_k=self.top_k)
        return [output_img]

    def postprocess(self, output_batch):
        """
        Create an image(jpeg) using the output tensor.
        """
        print("postprocess")
        postprocessed_data = []
        for op in output_batch:
            fp = BytesIO()
            op.save(fp, 'PNG')
            postprocessed_data.append(fp.getvalue())
            fp.close()
        return postprocessed_data

# if __name__ == "__main__":
#     handler = ModelHandler()
#     handler.initialize(None)
#     print("initialization is done. ")
#     preprocess_data = handler.preprocess([{"data": {"input_text": "a dog in water"}}])
#     print("preprocess data is done. ")
#     output = handler.inference(preprocess_data)
#     print("inference is done. ")
#     post_output = handler.postprocess(output)
#     print("Generated and saved images")
