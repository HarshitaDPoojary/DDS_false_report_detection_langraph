# pip install torch transformers docling_core

from pathlib import Path
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument

HERE = Path(__file__).resolve().parent
IMG_DIR = HERE / "images"
OUT_DIR = HERE / "Out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32  # bf16 optional on newer GPUs

processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
model = AutoModelForImageTextToText.from_pretrained(
    "ds4sd/SmolDocling-256M-preview",
    dtype=DTYPE,
    attn_implementation="sdpa",  # <-- avoids flash_attn
).to(DEVICE)

def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in EXTS and p.is_file()])

def docling_for_image(image):
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": "Convert this page to docling."}]
    }]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=2048)  # 8192 can be heavy
    prompt_len = inputs["input_ids"].shape[1]
    trimmed = generated_ids[:, prompt_len:]
    doctags = processor.batch_decode(trimmed, skip_special_tokens=False)[0].lstrip()

    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
    doc = DoclingDocument.load_from_doctags(doctags_doc, document_name="Document")
    return doctags, doc

def main():
    if not IMG_DIR.exists():
        print(f"[ERROR] Images folder not found: {IMG_DIR}")
        return
    paths = list_images(IMG_DIR)
    if not paths:
        print(f"[INFO] No images found in {IMG_DIR}")
        return

    print(f"[INFO] Processing {len(paths)} images from {IMG_DIR.resolve()}")
    for p in paths:
        try:
            image = load_image(str(p))  # local path ok
            doctags, doc = docling_for_image(image)

            md_path = OUT_DIR / f"{p.stem}.md"
            md_path.write_text(doc.export_to_markdown(), encoding="utf-8")

            # Optional HTML:
            # html_path = OUT_DIR / f"{p.stem}.html"
            # doc.save_as_html(html_path)

            print(f"{p.name}: OK -> {md_path.name}")
        except Exception as e:
            print(f"{p.name}: [ERROR] {e}")

if __name__ == "__main__":
    main()
