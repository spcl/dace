# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
"""
Parse the authors file and print for CITATION.cff
"""
with open("AUTHORS", "r") as f:
  content = f.readlines()

for i, l in enumerate(content[4:]):
    if l == "\n":
        end_idx = i + 4
        break
else:
    raise ValueError()

for author in content[4:end_idx]:
    names = author.strip().split()
    first_name = names[0]
    last_names = ' '.join(names[1:])
    text = f"- family-names: {last_names}\n  given-names: {first_name}"
    print(text)


