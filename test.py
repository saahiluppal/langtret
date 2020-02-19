with open("detect.cfg") as file:
    vals = file.readlines()

remove_linebreaks = lambda x: x.replace('\n','').strip()
remove_empty = lambda x: len(x) > 0

config = list(filter(remove_empty, map(remove_linebreaks, vals)))

for con in config:
    print(con)