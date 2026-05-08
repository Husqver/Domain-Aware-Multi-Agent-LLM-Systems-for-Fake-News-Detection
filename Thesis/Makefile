# EDIT HERE!
TEXMAIN=master.tex
# THAT'S IT :)

LATEXCMD = lualatex
BIBTEXCMD = biber

BASE = $(basename $(TEXMAIN))
PDF = $(BASE).pdf
GITREF=gitref.tex

ifeq ($(BINPATH),)
BINPATH = bin
endif

SVGS = $(shell find figures -name "*.svg")
SVGPDFS = $(SVGS:%.svg=%.pdf)

.PHONY: $(GITREF) all pdf touch clean texclean check figures

all: hello $(GITREF) pdf bye

hello::
	@echo "#####################################"
	@echo "### Hi, you look gorgeous today!  ###" 
	@echo "#####################################"

bye::
	@echo "###########################"
	@echo "### Have a nice day :)  ###" 
	@echo "###########################"

stamp: 
	for sm in $(SUBMODULES); do \
	  cd $(SUBMODULE_ROOT)/$$sm; \
	  make -s; \
	  cd - >/dev/null; \
	done
	date > stamp

pdf: $(PDF)

touch:
	touch $(TEXMAIN)

clean: texclean
	rm -f $(PDF)

texclean:
	rm -f *.run.xml *.acn *.acr *.alg *.bcf *.glg *.ist *.bbl *.blg *.cb *.cb2 *.gls *.glo *.aux *.log *.toc *.out *.ilg *.tdo *.lof *.lot $(GITREF)

distclean: clean
	rm -v stamp

$(PDF): $(TEXMAIN) figures $(TEXINCLUDE)
	$(LATEXCMD) $(TEXMAIN)
	$(LATEXCMD) $(TEXMAIN)
	-test -f $(BASE).glo  && makeindex $(BASE).glo -s nomencl.ist -o $(BASE).gls; true
	$(BIBTEXCMD) $(BASE); true
	-test -f $(BASE).idx && makeindex $(BASE); true
	$(LATEXCMD) $(TEXMAIN)
	$(LATEXCMD) $(TEXMAIN)

$(GITREF):
	git log -1 --date=short --format="format:\\gdef\\GITAbrHash{%h} \\gdef\\GITAuthorDate{%ad}" . > $(GITREF)

check:
	lacheck $(TEXMAIN)

figures: $(SVGPDFS)

figures/%.pdf: figures/%.svg
	$(BINPATH)/svg2pdf $<
