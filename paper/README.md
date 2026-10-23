# Paper

`main.tex` is the camera-ready ICAIF paper and `main.pdf` the compiled version.
The four figures it includes live in `paper/figures/`; every cited number comes
from the committed tables under `results/`.

Build locally with:

```bash
cd paper
latexmk -pdf main.tex   # or: tectonic main.tex
```

`acmart.cls` (v2.20, 2026/08/16) and `ACM-Reference-Format.bst` are vendored here
so the ACM rights block (`\setcopyright{cc}`) builds with older TeX bundles.
