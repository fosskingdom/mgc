# !pip install gradio
# !pip install transformers
# !pip install librosa
# !pip install torch
# !pip install torchaudio
# !pip install transformers

from gradio import Audio, Interface, Label
from transformers import pipeline

model_id = "sanchit-gandhi/distilhubert-finetuned-gtzan"
pipe = pipeline("audio-classification", model=model_id)


def classify_audio(filepath):
    preds = pipe(filepath)
    outputs = {}
    for p in preds:
        outputs[p["label"]] = p["score"]
    return outputs


gradio_app = Interface(
    fn=classify_audio,
    inputs=Audio(type="filepath", label="Upload Audio"),
    outputs=Label(label="Predicted Genre"),
    title="Music Genre Classification",
)
gradio_app.launch(share=True)
