TEX := mean_polynomials_main.tex
PDF := mean_polynomials_main.pdf
LATEX ?= pdflatex
LATEX_FLAGS ?= -interaction=nonstopmode

.PHONY: all pdf once clean

all: pdf

pdf:
	$(LATEX) $(LATEX_FLAGS) $(TEX)
	$(LATEX) $(LATEX_FLAGS) $(TEX)

once:
	$(LATEX) $(LATEX_FLAGS) $(TEX)

clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
