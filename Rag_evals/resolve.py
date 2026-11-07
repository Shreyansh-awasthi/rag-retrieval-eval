import re

def read(p):
    return open(p, encoding="utf-8").read().replace("\r\n", "\n")

def keep_both(p):
    seen, out = set(), []
    for line in read(p).split("\n"):
        if re.match(r"^(<<<<<<<|=======|>>>>>>>)", line):
            continue
        key = line.strip().lower()
        if key and key in seen:
            continue
        seen.add(key)
        out.append(line)
    open(p, "w", encoding="utf-8", newline="\n").write("\n".join(out))

def keep_ours(p):
    s = read(p)
    s = re.sub(r"<<<<<<< HEAD\n(.*?)=======\n.*?>>>>>>> [^\n]*\n?", r"\1", s, flags=re.S)
    open(p, "w", encoding="utf-8", newline="\n").write(s)

keep_both(".gitignore")
keep_both("requirements.txt")
keep_ours("README.md")
print("done")