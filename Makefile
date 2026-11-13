.NOTPARALLEL:

TEX_SOURCES = $(shell find . -name "*.tex")
BIB_SOURCES = $(shell find . -name "*.bib")

MAIN = main

default: $(MAIN).pdf

all: $(MAIN).pdf 

$(MAIN).pdf: $(TEX_SOURCES) $(BIB_SOURCES) ./figures/delta_t_results.pdf  ./figures/data_parallel_results.pdf ./figures/relative_error_vs_dynamic.pdf ./figures/multi_host_ranking.pdf Makefile
	pdflatex $(MAIN)
	bibtex $(MAIN)
	pdflatex $(MAIN)
	pdflatex $(MAIN)

./figures/delta_t_results.pdf: ./figures/results_delta_t_lower_bound.json ./figures/results_delta_t_upper_bound.json ./figures/plot_delta_t_results.py
	cd figures; python3 ./plot_delta_t_results.py --lower-bound-result-file ./results_delta_t_lower_bound.json --upper-bound-result-file ./results_delta_t_upper_bound.json --output-pdf ./delta_t_results.pdf

./figures/data_parallel_results.pdf: ./figures/results_data_parallel.json ./figures/plot_data_parallel_results.py
	cd figures; python3 ./plot_data_parallel_results.py ./data_parallel_results.json

./figures/relative_error_vs_dynamic.pdf: ./figures/results_single_host_relative_error.json ./figures/plot_single_host_relative_errors.py 
	cd figures; python3 ./plot_single_host_relative_errors.py --json-result-file ./results_single_host_relative_error.json --output-pdf relative_error_vs_dynamic.pdf

./figures/multi_host_ranking.pdf: ./Makefile ./figures/results_multi_host_ranking.json ./figures/plot_multi_host_ranking.py
	cd figures; python plot_multi_host_ranking.py \
    ./results_multi_host_ranking.json \
    --output multi_host_ranking.pdf \
    --bar-height 0.035 \
    --row-gap 0.045 \
    --group-gap 0.04 \
    --y-margin 0.02 \
    --header-gap 0.05  \
    --vertical-scale 0.45 \
    --broken-begin 330 \
    --broken-end 535 \
    --broken-gap 0.05 \
    --xtick-step 100


clean:
	/bin/rm -rf *.aux *log *.blg *.bbl *.out *.toc *.lof *.lot comment.cut $(MAIN).pdf
