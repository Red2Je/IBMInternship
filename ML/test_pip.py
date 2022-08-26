import os
l = [] 
with os.popen("pip list") as f:
    l.append(f.readlines())

print(l)