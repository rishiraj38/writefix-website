"""Build the site: inline engine.js into src.html -> index.html (served by GitHub Pages) and page.html (Claude artifact copy)."""
import os
R = os.path.dirname(os.path.abspath(__file__))
s = open(os.path.join(R, "src.html")).read().replace("/*ENGINE*/", open(os.path.join(R, "engine.js")).read())
open(os.path.join(R, "page.html"), "w").write(s)
head = '<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n<style>body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n'
t = s.index('<canvas id="field"')
open(os.path.join(R, "index.html"), "w").write(head + s[:t] + "</head><body>\n" + s[t:] + "\n</body></html>\n")
print("built index.html and page.html")
