
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchcam.methods import GradCAM
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64
from fpdf import FPDF
import pyttsx3 

st.set_page_config(page_title="🩻 AI Chest X-Ray Classifier", page_icon="🫁", layout="wide")

background_path = "/Users/abhisamadhiya/Desktop/FLUFFYCHEST/download.jpeg"

def set_bg_from_local(image_path):
    with open(image_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
        div.stButton > button {{
            background: linear-gradient(135deg, #00B4D8, #0077B6);
            color: white; border-radius: 10px;
            padding: 0.6em 1.2em; font-weight: 600;
            border: none; transition: 0.3s;
        }}
        div.stButton > button:hover {{
            background: linear-gradient(135deg, #0077B6, #023E8A);
            transform: scale(1.05);
        }}
        h1, h2, h3 {{ color: #CAF0F8 !important; text-shadow: 1px 1px 3px #000; }}
        .stMarkdown, .stText {{ color: white !important; }}
        .stAlert {{ background-color: rgba(0,0,0,0.6); border-radius: 10px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

set_bg_from_local(background_path)

st.markdown("<h1 style='text-align:center;'>🩻 Hybrid AI Chest X-Ray Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px; color:#CAF0F8;'>Fusion of CNN + Swin Transformer + Voice Feedback</p>", unsafe_allow_html=True)

WEIGHTS_PATH = "/Users/abhisamadhiya/Desktop/FLUFFYCHEST/hybrid_fusion_gradcam_best.pth"
NUM_CLASSES = 6
CLASS_NAMES = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Pneumonia"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HybridFusionModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.cnn = timm.create_model("resnet18", pretrained=False, features_only=True)
        self.swin = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, features_only=True)
        self.fusion_conv = None
        self.classifier = None

    def _init_fusion(self, cnn_feat, swin_feat):
        total = cnn_feat.shape[1] + swin_feat.shape[1]
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, NUM_CLASSES),
        )

    def forward(self, x):
        cnn_feats = self.cnn(x)[-1]
        swin_feats = self.swin(x)[-1]
        if swin_feats.shape[2:] != cnn_feats.shape[2:]:
            swin_feats = F.interpolate(swin_feats, size=cnn_feats.shape[2:], mode="bilinear")
        if self.fusion_conv is None:
            self._init_fusion(cnn_feats, swin_feats)
        fused = torch.cat((cnn_feats, swin_feats), dim=1)
        fused = self.fusion_conv(fused)
        out = self.classifier(fused)
        return out, fused

@st.cache_resource
def load_model():
    model = HybridFusionModel(num_classes=NUM_CLASSES).to(DEVICE)
    _ = model(torch.randn(1, 3, 224, 224).to(DEVICE))
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()
st.sidebar.success("✅ Model loaded successfully")

uploaded_file = st.file_uploader("📤 Upload a Chest X-Ray", type=["jpg", "jpeg", "png"])
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img, caption="🩻 Uploaded X-Ray", use_container_width=True)

    input_tensor = transform(img).unsqueeze(0).to(DEVICE)
    cam_extractor = GradCAM(model, target_layer="fusion_conv.2")

    out, _ = model(input_tensor)
    probs = torch.softmax(out, dim=1)[0]
    pred_class = probs.argmax().item()
    disease = CLASS_NAMES[pred_class]

    cam = cam_extractor(pred_class, out)[0]
    if cam.dim() == 2:
        cam = cam.unsqueeze(0).unsqueeze(0)
    elif cam.dim() == 3:
        cam = cam.unsqueeze(0)
    elif cam.dim() == 1:
        cam = cam.view(1, 1, int(cam.shape[0] ** 0.5), int(cam.shape[0] ** 0.5))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cam.squeeze()

    fig, ax = plt.subplots()
    ax.imshow(to_pil_image(input_tensor[0].cpu() * 0.5 + 0.5))
    ax.imshow(to_pil_image(cam.cpu()), cmap="jet", alpha=0.5)
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    with col2:
        st.image(buf, caption=f"🔥 Grad-CAM — {disease}", use_container_width=True)

    try:
        engine = pyttsx3.init("nsss")
        engine.setProperty('rate', 175)
        engine.say(f"Prediction complete. Detected condition is {disease}")
        engine.runAndWait()
    except Exception:
        st.warning("⚠️ Voice feedback unavailable on this system.")

    st.markdown("## 💡 Predicted Condition")
    st.success(f"### {disease}")

    with st.expander("📊 View Prediction Probabilities"):
        for cls, p in zip(CLASS_NAMES, probs):
            st.write(f"**{cls}** — {p.item()*100:.2f}%")
            st.progress(float(p))

    rec = {
        "Atelectasis": ("🏃‍♂️ Deep breathing, stay hydrated.", "💊 Salbutamol, Ambroxol, Paracetamol"),
        "Cardiomegaly": ("🥗 Low-sodium diet, avoid alcohol.", "💊 Enalapril, Metoprolol, Furosemide"),
        "Effusion": ("💧 Balanced fluids, gentle breathing.", "💊 Furosemide, Azithromycin"),
        "Infiltration": ("😴 Rest, steam therapy, eat healthy.", "💊 Amoxicillin, Ambroxol"),
        "Mass": ("🏥 Biopsy follow-up, avoid smoke.", "💊 Targeted therapy / steroids (doctor advised)"),
        "Pneumonia": ("💊 Complete antibiotics, hydrate, rest.", "💊 Amoxicillin, Levofloxacin, Paracetamol")
    }

    advice, meds = rec[disease]
    st.markdown("### 💡 Recovery Advice")
    st.info(advice)
    st.markdown("### 💊 Recommended Medicines")
    st.warning(meds)

else:
    st.info("👆 Upload a chest X-ray to begin diagnosis.")
