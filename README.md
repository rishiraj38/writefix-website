# WriteFix website

Landing page for the WriteFix prototype. The live demo runs the real model in the visitor's browser (`engine.js`, weights in `model/`).

Edit `src.html` or `engine.js`, then run `python3 build.py` to regenerate `index.html`.
New weights come from the main project: `python scripts/export_web.py <checkpoint>`.
