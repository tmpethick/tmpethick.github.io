


To generate:

```bash
brew install poppler

sh compile.sh
pdftoppm -png data.pdf data -r 300
pdftoppm -png trait_cause.pdf trait_cause -r 300
pdftoppm -png GRN.pdf GRN -r 300

```

... Then move to `/assets`.
