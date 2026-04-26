from pathlib import Path

p = Path(__file__).resolve().parent.parent / "static" / "index.html"
t = p.read_text(encoding="utf-8")
start = t.find('<link href="https://fonts.googleapis.com')
end = t.find("</style>") + len("</style>")
if start < 0 or end < 10:
    raise SystemExit(f"markers not found: start={start} end={end}")
new = """<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,400;0,600;0,700;1,400&display=swap" rel="stylesheet" />
<link rel="stylesheet" href="/static/brutalist.css" />
"""
p.write_text(t[:start] + new + t[end:], encoding="utf-8")
print("OK", p)
