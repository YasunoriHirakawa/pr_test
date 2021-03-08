import os
import argparse
from glob import iglob
from PIL import Image

parser = argparse.ArgumentParser(description="")
parser.add_argument("--type", type=str, default="reference")
args = parser.parse_args()
if args.type == "reference":
    default_directory = "default_references"
    saving_directory = "reference_images"
else:
    default_directory = "default_query"
    saving_directory = "query_images"

for i, path in enumerate(iglob(os.path.join(default_directory, "*"))):
    image = Image.open(path)
    image = image.crop((0, 0, 5375, 2688/3*2))###
    resized_image = image.resize((224*4, 224), Image.LANCZOS)
    os.makedirs(saving_directory, exist_ok=True)
    resized_image.save(f"{saving_directory}/{i}_resized.jpg")
