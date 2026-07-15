# Project 04 — Deploying the MobileNetV2 Model with Streamlit

## Overview

This project deploys the transfer-learning cat vs. dog classifier from **Project 03** as a live, interactive web application using **Streamlit**. Users can upload any image and receive a real-time prediction (cat or dog) with a confidence score — turning a trained model into a usable, shareable product.

**Live app:** [add your Streamlit Cloud URL here once deployed]
**Model used:** MobileNetV2 (transfer learning), trained in Project 03 — 98.5% validation accuracy

---

## What the App Does

1. User uploads a `.jpg`, `.jpeg`, or `.png` image.
2. The image is displayed back to the user for confirmation.
3. The image is preprocessed to match the model's expected input (160×160, MobileNetV2 normalization).
4. The trained model predicts cat or dog, along with a confidence percentage.
5. A confidence-based message gives the user context on how certain the model is:
   - ≥95% — "highly confident"
   - ≥80% — "reasonably confident"
   - <80% — "uncertain, try another clear image"

---

## Tech Stack

- **Streamlit** — web app framework, pure Python, no HTML/CSS/JS required
- **TensorFlow / Keras** — loads and runs the trained `.keras` model
- **Pillow (PIL)** — image handling and preprocessing
- **NumPy** — array operations for model input

---

## Project Structure

```
Project-04/
├── app.py                              # Streamlit application
├── project03_mobilenet_cats_dogs.keras # Trained model from Project 03
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## Deployment

This app is deployed on **Streamlit Community Cloud**, which provides a free, permanent, publicly accessible URL directly connected to this GitHub repository. Any future commit to `app.py` automatically redeploys the live app.

### How it was deployed:
1. Pushed `app.py`, the trained `.keras` model file, and `requirements.txt` to this repository.
2. Connected the repo at [share.streamlit.io](https://share.streamlit.io) using GitHub sign-in.
3. Selected this repository, branch `main`, and `app.py` as the entry point.
4. Streamlit Cloud installed dependencies from `requirements.txt` and deployed automatically.

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Series

This is the fourth project in a progressive computer vision series:

| Project | Focus |
|---|---|
| 01 | CNN from scratch — CIFAR-10 (10-class classification) |
| 02 | CNN from scratch — Cats vs Dogs (binary classification) |
| 03 | Transfer learning with MobileNetV2 — Cats vs Dogs (98.5% val. accuracy) |
| **04** | **Deploying the Project 03 model as a live web app** |

Together, these projects demonstrate the full applied ML lifecycle: building and training models from scratch, improving performance with transfer learning, and shipping a working product a non-technical user can actually interact with.
