DOT ?= dot
FIGURE_DIR := ../figures
SOURCES := algorithm_architecture training_workflow

.PHONY: all clean

all: $(foreach name,$(SOURCES),$(FIGURE_DIR)/$(name).png $(FIGURE_DIR)/$(name).pdf $(FIGURE_DIR)/$(name).svg)

$(FIGURE_DIR)/%.png: %.dot
	@mkdir -p $(FIGURE_DIR)
	$(DOT) -Gdpi=240 -Tpng $< -o $@

$(FIGURE_DIR)/%.pdf: %.dot
	@mkdir -p $(FIGURE_DIR)
	$(DOT) -Tpdf $< -o $@

$(FIGURE_DIR)/%.svg: %.dot
	@mkdir -p $(FIGURE_DIR)
	$(DOT) -Tsvg $< -o $@

clean:
	rm -f $(foreach name,$(SOURCES),$(FIGURE_DIR)/$(name).png $(FIGURE_DIR)/$(name).pdf $(FIGURE_DIR)/$(name).svg)
