# Conference Paper

**Author:** Xin Su

`main.tex` is the English paper version of the project. It uses a neutral
10-point, two-column conference layout because the competition has not supplied
a venue-specific LaTeX class. The affiliation field is omitted until final
submission metadata is available.

Build from this directory with:

```bash
make
```

The Makefile prefers `latexmk` and falls back to Tectonic. Figures are referenced
from `../docs/figures`, and citations are stored in `references.bib`.

Before submission:

1. Add the affiliation if required by the submission system.
2. Replace the document class and margin setup if the organizer publishes an
   official template; keep the paper body and BibTeX database unchanged.
3. Confirm the page limit and move the appendix to supplementary material if
   required.
4. Rebuild the PDF and inspect fonts, references, float placement, and page count.
