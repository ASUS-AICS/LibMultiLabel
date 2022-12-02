# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Variables for generating example codes 
QS_STEM := linear bert kimcnn
QS_CODES = $(addprefix examples/, $(addsuffix _quickstart.py, $(QS_STEM) ) )
DATASET_STEM := linear nn
DATASET_CODES = $(addprefix examples/dataset_example-, $(addsuffix .py, $(DATASET_STEM)) )
EXAMPLE_CODES = $(QS_CODES) $(DATASET_CODES)
COMPONENT_STEM := linear nn dataset
CODE_COMPONENTS = $(addprefix examples/, $(addsuffix _component.json, $(COMPONENT_STEM)) )
CODE_COMPONENTS_GENENRATORS = $(addprefix examples/, $(addsuffix _component.py, $(COMPONENT_STEM)) )

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile cli/flags.include $(EXAMPLE_CODES)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	rm -f cli/flags.include
	rm -f $(EXAMPLE_CODES) 

cli/flags.include:
	python3 cli/genflags.py > cli/flags.include

$(EXAMPLE_CODES): $(CODE_COMPONENTS)
	python3 examples/gen_examples.py --target $(basename $(notdir $@)) > $@

$(CODE_COMPONENTS): 
	@python3 $(CODE_COMPONENTS_GENENRATORS)




