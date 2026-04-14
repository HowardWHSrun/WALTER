# WALTER

**WALTER** = *Walking Alternating Tripod for Evolutionary Research* — a twelve-DOF hexapod (firmware, MuJoCo RL, CAD, and documentation).

**Authors (Engineering 90):** Howard Wang, Matthew Shiley  
**Advisor:** Allan Moser (Swarthmore College)

## Contents (repository map)

| Area | Path |
|------|------|
| Final report (PDF, LaTeX, figures) | [`final_report/`](final_report/) |
| Adaptive firmware and docs | [`adaptive_ondevice_hexapod/`](adaptive_ondevice_hexapod/) |
| MuJoCo PPO training code | [`hexapod_training_new/`](hexapod_training_new/) |
| Released checkpoints (zips) and eval helpers | [`e90_repo/`](e90_repo/) |
| Mid-semester presentation sources | [`Presentation/`](Presentation/) |

Training rollouts under `hexapod_training_new/runs/` are omitted from this repository to save space; regenerate with `train.py` and the configs in `hexapod_training_new/configs/`.

## Build the final report PDF

From `final_report/`:

```bash
pdflatex E90_Final_Report.tex
bibtex E90_Final_Report
pdflatex E90_Final_Report.tex
pdflatex E90_Final_Report.tex
```

A prebuilt `E90_Final_Report.pdf` is included.

## License

Add a `LICENSE` file if you want explicit terms; otherwise default GitHub copyright applies.
