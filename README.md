# BDD100K Risk Assessment VQA Generator  
Transforming BDD100K driving annotations into a **Visual Question Answering (VQA)** dataset for **driving safety and risk assessment**.

This system generates:  
- A question (e.g., *“Is it safe to proceed?”*)  
- A safety-based answer (*Yes / No / Slow Down*)  
- A natural-language rationale  
- Metadata (distance, traffic light color, etc.)

It uses **geometric reasoning** + **traffic signals** to infer driving risk.

---

# The Core Logic

The pipeline consists of three major components:

1. **Context Extraction (Distance Estimation)**  
2. **Proximity Classification**  
3. **Decision + Rationale Generation**

---

# 1. 🧠 Context Extraction (Distance Estimation)

The `VehicleDistanceEstimator` uses a simplified **Pinhole Camera Model** to estimate the distance to a vehicle using only bounding box size (no calibration required).

### 📌 Formula Used

The distance is estimated using:

\[
Z = \frac{K \cdot W_{\text{real}}}{w}
\]

### 🔍 Parameter Summary

| Parameter | Code Variable | Value | Description |
|----------|---------------|--------|-------------|
| \(Z\) | `distance_combined` | — | Estimated distance (meters) |
| \(K\) | `assumed_focal_length` | **1000 px** | Assumed focal length |
| \(W_{real}\) | `avg_vehicle_width` / `avg_vehicle_height` | **1.8 m**, **1.5 m** | Real-world vehicle size |
| \(w\) | `bbox_width` / `bbox_height` | — | Vehicle pixel size |

### 🚙 Selecting the Correct Vehicle

From all vehicles, the algorithm selects the one that is:

- **Closest**  
- **Centrally located** (within **10%** of image width)

This avoids picking vehicles in other lanes.

---

# 2. 📏 Proximity Classification

Distance \(Z\) is mapped to qualitative proximity:

| Proximity | Range | Variable |
|----------|--------|-----------|
| **Close** | \(Z \le 8.0\) m | `CLOSE_DISTANCE_THRESHOLD_M` |
| **Medium** | \(8.0 < Z \le 25.0\) m | `MEDIUM_DISTANCE_THRESHOLD_M` |
| **Far** | \(Z > 25.0\) m | — |

---

# 3. 🧮 Decision Logic (`decide_answer`)

The final safety recommendation is determined by:

- Vehicle proximity  
- Traffic light color  
- Randomly sampled question type  

### 🚦 Safety Rules

| Condition | Answer |
|----------|--------|
| **Traffic light = Red** OR **Vehicle = Close** | **No** (Stop) |
| **Traffic light = Yellow** OR **Vehicle = Medium** | **slow_down** |
| **Traffic light = Green** AND **Vehicle = Far/None** | **Yes** (Proceed) |

---

# 📝 Example VQA Output

```json
{
  "image": "/path/to/bdd100k/images/100k/val/0101b7e9-a5e2d67d.jpg",
  "question": "Is it safe to proceed?",
  "answer": "Slow Down, it is generally safe to proceed, but caution is advised because the traffic light is green and a medium vehicle is ahead.",
  "meta": {
    "front_vehicle_proximity": "medium",
    "front_vehicle_distance": 15.5,
    "traffic_light": "green",
    "weather": "clear",
    "timeofday": "day"
  }
}
