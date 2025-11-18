# BDD100K Risk Assessment VQA Dataset & BLIP-2 Fine-Tuning

## Objective & Overview

This project builds a **Risk Assessment Visual Question Answering (VQA)** dataset from the **BDD100K** autonomous-driving dataset.  
Instead of captions, the system generates **question–answer–rationale triplets** that capture driving safety and scene understanding.

The goal is to enable models such as **BLIP-2** to answer safety-critical questions like:

- *“Is it safe to proceed?”*  
- *“Should the car slow down?”*  
- *“Does the vehicle need to stop?”*  

The pipeline integrates:

- **Pinhole Camera Geometry** → Estimate distance to the front vehicle  
- **Label Reasoning** → Detect traffic lights, weather, scene context  
- **Rule-based Safety Logic** → Decide *Yes*, *No*, or *Slow Down*  
- **Rationale Generation** → Provide interpretable explanations  

This results in a high-quality VQA dataset for training **autonomous driving reasoning models**.

---

## Output Examples

Demo @
<a href="https://huggingface.co/spaces/<USERNAME>/<SPACE_NAME>">
  <img src="https://raw.githubusercontent.com/huggingface/branding/main/square-logo.svg" width="110"/>
</a>


### Example 1  
**Input Image:**  
<img src="https://github.com/MargiPandya27/BLIP2_Finetuned_BDD100k_VQA/blob/main/logs/demo1.png" width="400"/>

**Generated VQA Sample:**
```
- **Question**: Does the car need to stop?
- **Answer**: No, it is not safe to proceed because a close vehicle is ahead.
```

---

### Example 2  
**Input Image:**  
<img src="https://github.com/MargiPandya27/BLIP2_Finetuned_BDD100k_VQA/blob/main/logs/demo2.png" width="400"/>

**Generated VQA Sample:**
```
- **Question**: Is it safe to proceed?
- **Answer**: Yes, it is safe to proceed because the traffic light is green and no close vehicle is ahead.
```

---

## Model Details

The model is based on **BLIP-2 (OPT-2.7B)**.  
The training process:

1. First fine-tunes BLIP-2 on **BDD100K scene captioning** (https://github.com/MargiPandya27/Blip2_Finetuning_Scene_Captioning) 
2. Then adapts the model to the **VQA risk-assessment task** using **QLoRA**  
3. Produces a LoRA adapter that can be merged or loaded into BLIP-2 for inference

This two-stage approach improves scene understanding and reduces overfitting.

---

## Training

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training
```bash
python main.py
```

You can adjust all hyperparameters in train_args.py

Training Pipeline Includes:
This automatically:
- Downloads BDD100K via Kaggle Hub
- Processes JSON labels into semantic captions
- Splits data into train/val/test
- Loads BLIP-2 with LoRA configuration
- Trains for 5 epochs with:
  - Batch size: 4 per device
  - Learning rate: 5e-5
  - Gradient accumulation: 4 steps
  - FP16 precision


### 3. Output
Fine-tuned LoRA adapter is saved to ./blip2-finetuned-vqa/. If running on Colab, saving to Google Drive is recommended.

## Dataset Pipeline

A complete breakdown of the data processing stages is available in data_README.md (https://github.com/MargiPandya27/BLIP2_Finetuned_BDD100k_VQA/blob/main/data_README.md).


## References

- BLIP-2 (Salesforce): https://huggingface.co/Salesforce/blip2-opt-2.7b
- BDD100K Dataset: https://bdd100k.com/
- PEFT / LoRA: https://github.com/huggingface/peft
