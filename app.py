import re
import cv2
import easyocr
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Food Label Ingredient Extractor", page_icon="🥣", layout="wide")
st.title("🥣 Food Label Ingredient Extractor")
st.write("Upload a food-label photo. The app reads the text, detects ingredient sections, and lets you review the result before downloading it.")

@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(["en"], gpu=False)


def preprocess_image(image_bytes):
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file could not be read as an image.")

    height, width = image.shape[:2]
    if width < 1800:
        scale = min(1800 / width, 3.0)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def extract_text(image_bytes):
    processed_image = preprocess_image(image_bytes)
    reader = load_ocr_reader()
    lines = reader.readtext(processed_image, detail=0, paragraph=False, add_margin=0.05)
    return "\n".join(line.strip() for line in lines if line.strip())


INGREDIENT_HEADING = re.compile(
    r"""(?im)^\s*(?:
        ingredients? |
        ingredients?\s+(?:list|declaration|statement) |
        list\s+of\s+ingredients? |
        ingredient\s+composition |
        composition\s+of\s+(?:the\s+)?product |
        composition |
        contents |
        what(?:'|’)s\s+inside
    )\s*(?:\([^)\n]{0,50}\))?\s*[:\-–]?\s*""",
    re.VERBOSE,
)

STOP_HEADING = re.compile(
    r"""(?im)^\s*(?:
        nutritional?\s+information |
        nutrition\s+(?:facts|information|values) |
        allergen\s+(?:information|warning|advice) |
        allergy\s+information |
        contains\s+(?:milk|soy|soya|wheat|nuts|peanuts|allergens?) |
        may\s+contain |
        other\s+flavou?rs?\s+ingredients? |
        recommended\s+dosage |
        directions\s+for\s+use |
        storage\s+instructions |
        manufactured\s+by |
        marketed\s+by |
        batch\s*(?:no\.?|number) |
        date\s+of\s+(?:mfg\.?|manufacture) |
        best\s+before |
        mrp\b
    )""",
    re.VERBOSE,
)


def find_ingredient_sections(ocr_text):
    """Find likely ingredient sections while preserving section order."""
    text = (ocr_text or "").replace("\x0c", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    matches = list(INGREDIENT_HEADING.finditer(text))
    sections = []

    for index, match in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        tail = text[match.end():next_start]
        stop_match = STOP_HEADING.search(tail)
        end = match.end() + stop_match.start() if stop_match else next_start

        section_text = text[match.end():end]
        section_text = re.sub(r"^[\s:;,\-–]+", "", section_text)
        section_text = re.sub(r"\s+", " ", section_text).strip()

        if len(section_text) >= 3:
            heading = match.group(0).strip(" \t\r\n:-–") or "Ingredients"
            sections.append({"heading": heading, "text": section_text})

    unique = []
    seen = set()
    for section in sections:
        key = re.sub(r"\W+", "", section["text"].lower())
        if key and key not in seen:
            seen.add(key)
            unique.append(section)
    return unique


def split_ingredients(text):
    """Split commas/semicolons outside parentheses, preserving additive codes."""
    if not text:
        return []
    ingredients, current, depth = [], [], 0

    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)

        if char in ",;" and depth == 0:
            item = "".join(current).strip()
            if item:
                ingredients.append(item)
            current = []
        else:
            current.append(char)

    final_item = "".join(current).strip()
    if final_item:
        ingredients.append(final_item)

    cleaned = []
    for item in ingredients:
        item = re.sub(r"^\s*(?:[-•*]|\d+[.)])\s*", "", item)
        item = re.sub(r"\s+", " ", item).strip(" \t;:-–")
        if item:
            cleaned.append(item)
    return cleaned


def clear_results():
    for key in ("ocr_text", "ingredient_sections"):
        st.session_state.pop(key, None)


uploaded_file = st.file_uploader(
    "Upload a food-label image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Sharp, well-lit, straight-on photos usually give better results.",
)

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    left, right = st.columns(2)

    with left:
        st.subheader("Uploaded image")
        try:
            pil_image = ImageOps.exif_transpose(Image.open(uploaded_file))
            st.image(pil_image, width="stretch")
        except Exception:
            st.error("The image could not be displayed.")

    with right:
        st.subheader("Extract text")
        st.write("The app checks several common ingredient headings and can show multiple sections.")
        if st.button("Extract Ingredients", type="primary"):
            clear_results()
            try:
                with st.spinner("Reading the label with OCR..."):
                    ocr_text = extract_text(image_bytes)
                    sections = find_ingredient_sections(ocr_text)
                st.session_state["ocr_text"] = ocr_text
                st.session_state["ingredient_sections"] = sections
            except Exception as error:
                st.error("Text extraction failed. Check the image and try again.")
                st.exception(error)

    if "ocr_text" in st.session_state:
        sections = st.session_state.get("ingredient_sections", [])
        st.divider()
        st.subheader("Ingredient section")

        if sections:
            options = [
                f"{i + 1}. {s['heading']}: {s['text'][:75]}{'...' if len(s['text']) > 75 else ''}"
                for i, s in enumerate(sections)
            ]
            selected_index = st.selectbox(
                "Choose the ingredient section",
                options=list(range(len(options))),
                format_func=lambda i: options[i],
            )
            selected_text = sections[selected_index]["text"]
            st.caption("Review and correct the OCR text before downloading; OCR can make mistakes.")
            edited_text = st.text_area("Review or correct the ingredient text", value=selected_text, height=140)
            ingredients = split_ingredients(edited_text)

            st.subheader("Extracted ingredients")
            if ingredients:
                st.success(f"Found {len(ingredients)} ingredient entries.")
                for number, ingredient in enumerate(ingredients, start=1):
                    st.write(f"{number}. {ingredient}")
                download_text = "\n".join(f"{i}. {item}" for i, item in enumerate(ingredients, start=1))
                st.download_button("Download ingredients (.txt)", download_text, "extracted_ingredients.txt", "text/plain")
            else:
                st.warning("No ingredient entries could be split. Edit the text above and try again.")
        else:
            st.warning("No recognized ingredient heading was found. This does not mean the label has no ingredients.")
            manual_text = st.text_area(
                "Paste or type the ingredient section here",
                value="",
                height=140,
                placeholder="Example: Soya Protein, Maltodextrin, Corn Starch...",
            )
            ingredients = split_ingredients(manual_text)
            if ingredients:
                st.subheader("Ingredients from your corrected text")
                for number, ingredient in enumerate(ingredients, start=1):
                    st.write(f"{number}. {ingredient}")
                download_text = "\n".join(f"{i}. {item}" for i, item in enumerate(ingredients, start=1))
                st.download_button(
                    "Download ingredients (.txt)",
                    download_text,
                    "extracted_ingredients.txt",
                    "text/plain",
                    key="manual_download",
                )

        with st.expander("View complete OCR text"):
            st.text(st.session_state.get("ocr_text", ""))
