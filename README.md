# 🧭 PYAXIOMA-JH: Non-Contact Angular Measurement via Computer Vision

Welcome to **PYAXIOMA-JH** — the *Python Axis of Magnetic Oscillation and Motion Analysis*, developed by Joseph Havens at the University of Kansas. This project offers a lightweight, affordable, and open-source method for real-time angular displacement tracking using computer vision techniques — requiring only a standard webcam, a rotating object, and marker-based segmentation.

---

## 🌌 Overview

**PYAXIOMA-JH** was created to solve a simple yet persistent problem in experimental physics labs:  
How can we **track angular motion** of a delicate rotating system **without physical sensors**, using just a camera and Python?

This repository demonstrates that it's not only possible — it's accurate, extensible, and visually insightful.

The full methodology is detailed in the paper:  
📝 *Proposed Methodology for Non-Contact Angular Measurement via Computer Vision – PYAXIOMA-JH*

---

## 📁 Repository Structure
PYAXIOMA-JH/

├── raw_data/           # All raw measurement CSVs (mine and Lucciana’s)

├── data_analysis/      # Interpolation, data merging, plotting, and confidence analysis

├── legacy_versions/    # Old PYAXIOMA builds and OpenCV experiments

├── pyaxioma_jh.py      # Final, functional PYAXIOMA-JH software

├── paper/              # LaTeX source for the final report (optional)

├── README.md           # This file!

## 🔬 Features

- 🎯 **Real-time angular tracking** with marker-based computer vision
- 🌀 **Optical flow motion masks** with dynamic motion contour selection
- 🧠 **Composite confidence estimation** based on stability, jerk, and trajectory features
- 🧰 **Full data logging** with timestamps, angle values, and frame confidence
- 🧩 **Multi-camera compatibility** for stereo-based angular interpolation
- 💻 Built entirely in **Python** with **OpenCV**, **NumPy**, and **Matplotlib**

---

## 🖼️ Sample Output

![Tracking Example](./figures/positive_tracking.png)  
*Marker vector, angle overlay, and motion mask visualized during tracking.*

---

## 📊 Using the Data

All raw angular data is stored as `.csv` files, one per amperage level.  
Each file includes:
- Time elapsed
- UTC timestamp
- Computed angle (in degrees)
- Frame-by-frame confidence score (0–100%)

The `data_analysis/` folder contains scripts for:
- Plotting angular displacement
- Calculating moving averages
- Interpolating multi-camera data streams
- Visualizing confidence breakdowns

---

## 📚 Citation

If you use or reference this project, please cite the accompanying paper:

> Havens, J. & Caceres Holgado, L. (2025). *Proposed Methodology for Non-Contact Angular Measurement via Computer Vision – PYAXIOMA-JH*. University of Kansas.

Or use the BibTeX entry.

---

## 🪪 License

This project is licensed under the **MIT License**.  
It allows for reuse, modification, and distribution, even in commercial contexts — as long as attribution is given.

---

## 👋 Acknowledgments

Special thanks to:
- **Lucciana Caceres Holgado**, for contributing the complementary camera system and data
- **Jessy Changstrom** and **Cole Douglas Le Mahieu**, for supporting the development effort
- The **University of Kansas Department of Physics and Astronomy**

---

## 🛠️ Dependencies

- `opencv-python`
- `numpy`
- `matplotlib`
- `pandas`

Install them with:

`pip install -r requirements.txt`

✨ Future Work
	•	Stereo triangulation of angular data using synced dual-camera interpolation
	•	Improved real-time confidence prediction using CNN-based motion classifiers
	•	GUI-based parameter tuning and live result visualization

⸻

Built with curiosity, coffee, and a cue ball.

---
