from __future__ import annotations

import base64
import os

import groundingdino.datasets.transforms as T
import numpy as np
import openai
import requests
import torch
import torchvision
from groundingdino.util.inference import load_model, predict
from PIL import Image
from scipy import spatial
from torchvision import transforms
from torchvision.ops import box_convert
from transformers import (Blip2ForConditionalGeneration, Blip2Processor,
                          SamModel, SamProcessor)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))

device = "cuda" if torch.cuda.is_available() else "cpu"
sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

## Grounding Dino settings
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25
gd_model = load_model(
    os.path.join(BASE_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
    os.path.join(BASE_PATH, "GroundingDINO/weights/groundingdino_swint_ogc.pth"))

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model_blip = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.bfloat16
)

sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

api_file = os.path.join(BASE_PATH, 'api.key')

with open(api_file) as f:
    api_key = f.readline().splitlines()
openai.api_key = api_key[0]
EMBEDDING_MODEL = "text-embedding-ada-002"


model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

depth_transform = midas_transforms.dpt_transform

def convert_bbox(image_source: torch.Tensor, boxes: torch.Tensor) -> np.ndarray:
    _,h, w = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    return xyxy

def img_unormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(1)
    var = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(1)
    img = img*var + mean
    return torch.tensor(img*255, dtype=torch.uint8)

def image_transform(image):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333), # can we make this deterministic
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_transformed, _ = transform(image, None)
    return image_transformed


relatedness_fn = lambda x, y: 1 - spatial.distance.cosine(x, y)

class ImagePatch:
    """A Python class containing a crop of an image centered around a particular object, as well as relevant
    information.
    Attributes
    ----------
    cropped_image : array_like
        An array-like of the cropped image taken from the original image.
    left : int
        An int describing the position of the left border of the crop's bounding box in the original image.
    lower : int
        An int describing the position of the bottom border of the crop's bounding box in the original image.
    right : int
        An int describing the position of the right border of the crop's bounding box in the original image.
    upper : int
        An int describing the position of the top border of the crop's bounding box in the original image.

    Methods
    -------
    find(object_name: str)->List[ImagePatch]
        Returns a list of new ImagePatch objects containing crops of the image centered around any objects found in the
        image matching the object_name.
    exists(object_name: str)->bool
        Returns True if the object specified by object_name is found in the image, and False otherwise.
    verify_property(property: str)->bool
        Returns True if the property is met, and False otherwise.
    best_text_match(option_list: List[str], prefix: str)->str
        Returns the string that best matches the image.
    simple_query(question: str=None)->str
        Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?".
    compute_depth()->float
        Returns the median depth of the image crop.
    crop(left: int, lower: int, right: int, upper: int)->ImagePatch
        Returns a new ImagePatch object containing a crop of the image at the given coordinates.
    """

    def __init__(self, image: Image.Image | torch.Tensor | np.ndarray, left: int | None = None, lower: int | None = None,
                 right: int | None = None, upper: int | None = None, parent_left=0, parent_lower=0, queues=None,
                 parent_img_patch=None, mask = None):
        """Initializes an ImagePatch object by cropping the image at the given coordinates and stores the coordinates as
        attributes. If no coordinates are provided, the image is left unmodified, and the coordinates are set to the
        dimensions of the image.

        Parameters
        -------
        image : array_like
            An array-like of the original image.
        left : int
            An int describing the position of the left border of the crop's bounding box in the original image.
        lower : int
            An int describing the position of the bottom border of the crop's bounding box in the original image.
        right : int
            An int describing the position of the right border of the crop's bounding box in the original image.
        upper : int
            An int describing the position of the top border of the crop's bounding box in the original image.

        """

        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)
        elif isinstance(image, np.ndarray):
            image = torch.tensor(image).permute(1, 2, 0)
        elif isinstance(image, torch.Tensor) and image.dtype == torch.uint8:
            image = image / 255


        if left is None and right is None and upper is None and lower is None:
            self.cropped_image = image
            self.left = 0
            self.lower = 0
            self.right = image.shape[2]  # width
            self.upper = image.shape[1]  # height
            if mask == None:
                self.mask = torch.ones(self.cropped_image.shape[1],self.cropped_image.shape[2])
            else:
                self.mask = mask
        else:
            self.cropped_image = image[:, image.shape[1]-upper:image.shape[1]-lower, left:right]
            self.left = left + parent_left
            self.upper = upper + parent_lower
            self.right = right + parent_left
            self.lower = lower + parent_lower
            if mask == None:
                self.mask = torch.ones(self.cropped_image.shape[1],self.cropped_image.shape[2])
            else:
                self.mask = mask[image.shape[1]-upper:image.shape[1]-lower, left:right]

        self.height = self.cropped_image.shape[1]
        self.width = self.cropped_image.shape[2]

        # self.cache = {}
        self.queues = (None, None) if queues is None else queues

        self.parent_img_patch = parent_img_patch
        self.parent_mask = mask

        self.horizontal_center = (self.left + self.right) / 2
        self.vertical_center = (self.lower + self.upper) / 2

        if self.cropped_image.shape[1] == 0 or self.cropped_image.shape[2] == 0:
            raise Exception("ImagePatch has no area")

        if self.height<700 or self.width<700:
            scale = max(700/self.width,700/self.height)
            self.height = int(self.height*scale)
            self.width = int(self.width*scale)
            self.cropped_image = torchvision.transforms.Resize((self.height,self.width))(self.cropped_image)
            self.mask = torchvision.transforms.Resize((self.height,self.width))(self.mask.unsqueeze(0)).squeeze()

        # draw bounding box and fill color
        # self.PIL_img = img_unormalize(self.cropped_image)
        # transform this image to PIL image
        self.PIL_img = torchvision.transforms.ToPILImage()(self.cropped_image)
        self.original_img = torchvision.transforms.ToPILImage()(image)


    def find(self, object_name: str) -> list[ImagePatch]:
        """Returns a list of ImagePatch objects matching object_name contained in the crop if any are found.
        Otherwise, returns an empty list.
        Parameters
        ----------
        object_name : str
            the name of the object to be found

        Returns
        -------
        List[ImagePatch]
            a list of ImagePatch objects matching object_name contained in the crop
        """
        input_image = image_transform(self.PIL_img)
        boxes, _logits, _phrases = predict(
            model=gd_model,
            image=input_image,
            caption=object_name,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        # box = torch.tensor(convert_bbox(self.cropped_image,boxes))
        box = np.array(convert_bbox(self.cropped_image,boxes))
        obj_list = []
        for b in box:
            mask_box = b.tolist()
            input_boxes = [mask_box]  # 2D location of a window in the image
            inputs = sam_processor(self.PIL_img, input_boxes=[input_boxes], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = sam_model(**inputs)
            mask = sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
            )[0].squeeze(0).detach().cpu().sum(0)>1.5
            obj = self.crop(int(b[0]),self.height-int(b[3]),int(b[2]),self.height-int(b[1]),mask=mask)
            # if obj.exists(object_name):
            obj_list.append(obj)
        return obj_list

    def exists(self, object_name: str) -> bool:
        """Returns True if the object specified by object_name is found in the image, and False otherwise.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        """
        prompt = f"Question: Do you see a {object_name} in the image? Answer (Y/N):"

        img_path = "query_image.png"
        # self.PIL_img.save(img_path,"PNG")
        self.PIL_img.save(img_path,"PNG")
        with open(img_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        headers = {
          "Content-Type": "application/json",
          "Authorization": f"Bearer {openai.api_key}"
        }

        payload = {
          "model": "gpt-4-vision-preview",
          "messages": [
            {
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": prompt
                },
                {
                  "type": "image_url",
                  "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                  }
                }
              ]
            }
          ],
          "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_message = response.json()["choices"][0]["message"]["content"]

        if response_message == 'Y':
            out= True
        else:
            out= False
        return out

    def verify_property(self, object_name: str, attribute: str) -> bool:
        """Returns True if the object possesses the property, and False otherwise.
        Differs from 'exists' in that it presupposes the existence of the object specified by object_name, instead
        checking whether the object possesses the property.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        attribute : str
            A string describing the property to be checked.
        """

        prompt = f"Question: Are the {object_name} {attribute}? Answer:"
        inputs = processor(images=self.PIL_img, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.bfloat16)

        generated_ids = model_blip.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        if generated_text == 'yes':
            out= True
        else:
            out= False
        return out

    def simple_query(self, question: str):
        """Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?". The questions are about basic perception, and are not meant to be used for complex reasoning
        or external knowledge.
        Parameters
        -------
        question : str
            A string describing the question to be asked.
        """
        prompt = question
#         inputs = processor(images=self.PIL_img, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.bfloat16)

#         generated_ids = model_blip.generate(**inputs)
#         generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

#         return generated_text

        img_path = "query_image.png"
        self.PIL_img.save(img_path,"PNG")
        # self.original_img.save(img_path,"PNG")
        with open(img_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        headers = {
          "Content-Type": "application/json",
          "Authorization": f"Bearer {openai.api_key}"
        }

        payload = {
          "model": "gpt-4-vision-preview",
          "messages": [
            {
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": prompt
                },
                {
                  "type": "image_url",
                  "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low"
                  }
                }
              ]
            }
          ],
          "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_message = response.json()["choices"][0]["message"]["content"]
        return response_message

    def best_image_match(self, list_patches: list[ImagePatch], content: list[str], return_index: bool = False) -> \
            ImagePatch | int | None:
        """Returns the patch most likely to contain the content.
        Parameters
        ----------
        list_patches : List[ImagePatch]
        content : List[str]
            the object of interest<
        return_index : bool
            if True, returns the index of the patch most likely to contain the object

        Returns
        -------
        int
            Patch most likely to contain the object
        """

        if len(list_patches) == 0:
            return None

        patch_embeddings = []
        for patch in list_patches:
            inputs = processor(images=patch.PIL_img, return_tensors="pt").to(device="cuda", dtype=torch.bfloat16)
            generated_ids = model_blip.generate(**inputs)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            ## convert generated_text to embedding
            response = openai.Embedding.create(model=EMBEDDING_MODEL, input=generated_text)
            patch_embeddings.append(response)

        scores = torch.zeros(len(patch_embeddings))
        for cont in content:
            query_embedding = openai.Embedding.create(model=EMBEDDING_MODEL, input=cont)
            relatedness = [relatedness_fn(query_embedding["data"][0]["embedding"], embed["data"][0]["embedding"]) for embed in patch_embeddings]
            scores += torch.tensor(relatedness)
        scores = scores / len(content)

        scores = scores.argmax().item()  # Argmax over all image patches

        if return_index:
            return scores
        return list_patches[scores]

    def crop(self, left: int, lower: int, right: int, upper: int, mask) -> ImagePatch:
        """Returns a new ImagePatch containing a crop of the original image at the given coordinates.
        Parameters
        ----------
        left : int
            the position of the left border of the crop's bounding box in the original image
        lower : int
            the position of the bottom border of the crop's bounding box in the original image
        right : int
            the position of the right border of the crop's bounding box in the original image
        upper : int
            the position of the top border of the crop's bounding box in the original image

        Returns
        -------
        ImagePatch
            a new ImagePatch containing a crop of the original image at the given coordinates
        """
        # make all inputs ints
        left = int(left)
        lower = int(lower)
        right = int(right)
        upper = int(upper)
        return ImagePatch(self.cropped_image, left, lower, right, upper, self.left, self.lower, queues=self.queues,
                          parent_img_patch=self, mask = mask)


    def llm_query(self, question, object_name, context=None, long_answer=True, queues=None):
        """Answers a text question using GPT-3. The input question is always a formatted string with a variable in it.

        Parameters
        ----------
        query: str
            the text question to ask. Must not contain any reference to 'the image' or 'the photo', etc.
        """
        query = question.format(object_name)

        prompt = query
        img_path = "query_image.png"
        # self.PIL_img.save(img_path,"PNG")
        self.original_img.save(img_path,"PNG")
        with open(img_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        headers = {
          "Content-Type": "application/json",
          "Authorization": f"Bearer {openai.api_key}"
        }

        payload = {
          "model": "gpt-4-vision-preview",
          "messages": [
            {
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": prompt
                },
                {
                  "type": "image_url",
                  "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low"
                  }
                }
              ]
            }
          ],
          "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_message = response.json()["choices"][0]["message"]["content"]
        return response_message

    def bool_to_yesno(self, bool_answer: bool) -> str:
        """Returns a yes/no answer to a question based on the boolean value of bool_answer.
        Parameters
        ----------
        bool_answer : bool
            a boolean value

        Returns
        -------
        str
            a yes/no answer to a question based on the boolean value of bool_answer
        """
        return "yes" if bool_answer else "no"

    def compute_depth(self):
        """Returns the median depth of the image crop
        Parameters
        ----------
        Returns
        -------
        float
            the median depth of the image crop
        """
        img = np.array(self.PIL_img)


        input_batch = depth_transform(img).to(device)
        with torch.no_grad():
            depth_map  = midas(input_batch)
            depth_map  = torch.nn.functional.interpolate(
                depth_map .unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_map = -depth_map+depth_map.max()
        return int(depth_map.median())  # Ideally some kind of mode, but median is good enough for now


    # def get_original_boxes(self):
    #     [[dog_patch.left,image_patch.height-dog_patch.upper,dog_patch.right,image_patch.height-dog_patch.lower]]


    # def get_mask(self, object_name: str, box: list):
    #     """Returns a mask of the object in question
    #     Parameters
    #     -------
    #     object name : str
    #         A string describing the name of the object to be masked in the image.
    #     object name : list
    #         Optional list of bounding box values of object.
    #     >>> # Generate the mask of the kid raising their hand
    #     >>> def execute_command(image) -> str:
    #     >>>     image_patch = ImagePatch(image)
    #     >>>     kid_patches = image_patch.find("kid")
    #     >>>     for kid_patch in kid_patches:
    #     >>>         if kid_patch.verify_property("kid", "raised hand"):
    #     >>>             return image_patch.get_mask("kid",[[kid_patch.left,image_patch.height-kid_patch.upper,kid_patch.right,image_patch.height-kid_patch.lower]])
    #     >>>     return None
    #     """
    #     return masks