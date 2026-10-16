import streamlit as st
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

st.set_page_config(page_title="VeriLens News Consistency", page_icon="🔎", layout="wide")
st.title("VeriLens: Multimodal News Verification")
st.write("Research prototype for checking semantic alignment between a headline and its image.")

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(name)
    model = CLIPModel.from_pretrained(name).to(device)
    model.eval()
    return processor, model, device

def similarity(text, image, processor, model, device):
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    t, im = out.text_embeds, out.image_embeds
    t = t / t.norm(dim=-1, keepdim=True)
    im = im / im.norm(dim=-1, keepdim=True)
    return float((t @ im.T).item())

headline = st.text_area("News headline or short text")
upload = st.file_uploader("Upload associated image", type=["png","jpg","jpeg","webp"])
threshold = st.slider("Similarity threshold (heuristic)", 0.0, 1.0, 0.25, 0.01)
if st.button("Analyze consistency", type="primary"):
    if not headline.strip() or upload is None:
        st.warning("Enter text and upload an image.")
    else:
        with st.spinner("Loading CLIP and analyzing..."):
            image = Image.open(upload).convert("RGB")
            processor, model, device = load_model()
            score = similarity(headline.strip(), image, processor, model, device)
        c1, c2 = st.columns(2)
        c1.image(image, caption="Uploaded image", use_container_width=True)
        c2.metric("Cosine similarity", f"{score:.4f}")
        c2.metric("Scaled score (not probability)", f"{score*100:.1f}/100")
        c2.write("Heuristic result: **Likely consistent**" if score >= threshold else "Heuristic result: **Possible mismatch**")
        st.info("This is semantic similarity, not a truth/fakeness probability. Validate the threshold on a labeled dataset.")

with st.expander("About and limitations"):
    st.write("Uses pretrained OpenAI CLIP embeddings and cosine similarity. A high score does not prove the news is true; a low score does not prove it is fake. Thresholds are dataset-dependent.")
