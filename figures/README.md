# Figures

`*.pdf` are the vector originals used by the paper. Keep these in the LaTeX
build: they scale without loss and print cleanly.

`png/*.png` are 300 dpi raster renders of the same figures, for READMEs,
slides, and anywhere PDF will not display inline.

To regenerate the vector originals:

    python3 make_fig_pipeline.py     # fig_pipeline.pdf
    python3 make_fig0_splits.py      # fig0_splits.pdf
    python3 regenerate_figures.py    # fig1, fig2, fig3 from results/

To re-render the PNGs from the PDFs:

    for f in fig*.pdf; do
      pdftoppm -png -r 300 -singlefile "$f" "png/${f%.pdf}"
    done
