PAPER := main

.PHONY: all clean

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf -interaction=nonstopmode -halt-on-error $(PAPER).tex; \
	elif command -v tectonic >/dev/null 2>&1; then \
		tectonic -X compile $(PAPER).tex; \
	else \
		echo "Install latexmk or tectonic to build the paper." >&2; \
		exit 1; \
	fi

clean:
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -C $(PAPER).tex; \
	else \
		rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log \
			$(PAPER).out $(PAPER).run.xml $(PAPER).synctex.gz $(PAPER).toc; \
	fi
