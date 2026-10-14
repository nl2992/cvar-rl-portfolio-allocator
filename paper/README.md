# Paper

`main.tex` is the camera-ready ICAIF paper and `main.pdf` the compiled version.
The four figures it includes live in `paper/figures/`; every cited number comes
from the committed tables under `results/`.

Build locally with:

```bash
cd paper
latexmk -pdf main.tex   # or: tectonic main.tex
```
