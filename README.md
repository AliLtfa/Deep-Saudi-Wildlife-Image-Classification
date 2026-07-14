# Deep Saudi Wildlife — Image Classification

Two Google Colab notebooks for classifying wild animals found in Saudi Arabia's
environment. One is a convolutional neural network (CNN) trained from scratch on
a labeled image dataset; the other is a zero-shot classifier built on OpenAI's
CLIP model that needs no training at all.

> **Note:** This is a pre-LLM-era project completed in **2023**. The
> notebooks are kept as-is for reference and learning purposes.

---

## Contents

| Notebook | Approach | Training required |
|---|---|---|
| `DeepSaudiWildLifeCNN.ipynb` | Custom CNN (TensorFlow / Keras) | Yes |
| `ZeroShotExample2.ipynb` | Zero-shot classification (OpenAI CLIP) | No |

---

## Dataset

An image library of wild animals found in Saudi Arabia's environment + some more, compiled
from Google Images (**~2 GB**). Because the images were scraped from Google
Images, the dataset is provided **for research purposes only**.

📁 **Google Drive:** https://drive.google.com/drive/folders/154-jEbFwrgbqqSiDd7F4rz47drekCevu?usp=sharing

The dataset is organized as one folder per animal class. The folder name is used
directly as the label:

```
Sample/
├── Arabian leopard/
│   ├── img001.jpg
│   └── ...
├── Sand cat/
│   ├── img001.jpg
│   └── ...
└── ...
```

---

## 1. CNN Classifier — `DeepSaudiWildLifeCNN.ipynb`

A convolutional neural network trained on the image dataset, where each Drive
folder becomes a class and its name is linked to the model's output label ID.

**Pipeline:**
1. Mount Google Drive and walk the dataset folders to collect image paths + labels.
2. Preview a class to confirm the data loaded correctly.
3. Data augmentation and preprocessing via `ImageDataGenerator`
   (rescale, shear, zoom, horizontal flip); images resized to **128×128**.
4. Build a Keras `Sequential` CNN (`Conv2D` → `MaxPooling2D` → `Flatten` →
   `Dense` → `Dropout` → `Dense`/softmax).
5. Compile with the **Adam** optimizer and **categorical cross-entropy** loss.
6. Train with `model.fit`.
7. Export the model and save an index-to-label mapping for later use.

**Exports:**
- `animal_classification_model.pb` (or `.h5`) — for app/server deployment.
- `index_to_label.json` and `index_to_label.csv` — map class indexes to animal names.
- **TensorFlow Lite** conversion for mobile use.

> ⚠️ **Beginner-code caveats (left unchanged on purpose):** the final `Dense`
> layer is set to `7` units and the augmentation generator is reused as the
> validation set. Set the output units to your actual number of classes and use a
> separate validation split if you adapt this notebook.

---

## 2. Zero-Shot Classifier — `ZeroShotExample2.ipynb`

Zero-shot image classification assigns an image to categories the model was
**never explicitly trained on**. Instead of learning fixed image→label mappings,
CLIP is a multi-modal model trained on a large set of images paired with text
descriptions, so it can be queried with free-form text labels.

**Pipeline:**
1. Install `transformers` (and `datasets`).
2. Load the **`openai/clip-vit-large-patch14`** checkpoint via the
   `zero-shot-image-classification` pipeline.
3. Upload an image.
4. Provide `candidate_labels` — a list of animal names as free text.
5. Process the image + labels with `CLIPProcessor` and run inference.
6. Softmax the logits to get per-label scores, sorted best-first.
7. Look up the top prediction on **Wikipedia** and display it with `matplotlib`.

**Why the labels matter:** In the CNN approach the class names come from dataset
folders. Here, the wording of the prompts drives the result — much like prompting
a language model. For example, dropping the `Sand cat` label leaves the model at
~50% on a wild cat; adding it back pushes the prediction to ~99%.

---

## Running the notebooks

Both notebooks are ready to run in **Google Colab** — open either one and use the
"Open in Colab" badge at the top.

1. Copy the dataset from the Drive link above into your own Google Drive.
2. Update the dataset path in the notebook (e.g. `/content/gdrive/MyDrive/Sample`).
3. Run the cells top to bottom.

**Main dependencies:** `tensorflow` / `keras`, `transformers`, `torch`,
`torchvision`, `Pillow`, `numpy`, `matplotlib`, `wikipedia`. Colab includes most
of these; the notebooks `pip install` the rest.

---

## License & Usage

The dataset is compiled from Google Images and is provided for **research
purposes only**. Please respect the original image sources and applicable
copyright when reusing any material.
