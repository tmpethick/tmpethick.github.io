# Personal blog

This is the source code for my personal blog.

## Develop

```
jupyter-book config sphinx .
cat conf_postfix.py >> conf.py
sphinx-build . _build -b dirhtml
```

For auto rebuild (mind that changes to `conf.py` will not be registered):

```
jupyter-book config sphinx .
cat conf_postfix.py >> conf.py
sphinx-autobuild . _build -b dirhtml
```

For complete rebuild:

```
rm -rf _build
jupyter-book config sphinx .
cat conf_postfix.py >> conf.py
sphinx-build . _build -b dirhtml
```

## Deploy

```
jupyter-book config sphinx .
cat conf_postfix.py >> conf.py
ablog build
ablog deploy
```
