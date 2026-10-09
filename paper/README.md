# Manuscript files

- `manuscript.tex`: manuscript source.
- `manuscript.pdf`: current PDF for reading and sharing.
- `build_xelatex/`: XeLaTeX output, including the compiled PDF and auxiliary files.
- `guide/`: journal submission guidance.
- `submission/`: prepared submission materials.

Run the following commands from `paper/`:

```bash
latexmk -interaction=nonstopmode -halt-on-error manuscript.tex
cp build_xelatex/manuscript.pdf manuscript.pdf
```

The local `.latexmkrc` selects XeLaTeX and directs compilation output to
`build_xelatex/`. Copy the PDF only after a successful build. Auxiliary files
at the top level of `paper/` are not needed.
