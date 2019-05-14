def progress(percentage, maxLen=30, line=1, eol="<br>"):
    if line < 1:
        line = 1
    str = ""
    for i in range(maxLen):
        str += '>'
    str += eol

    for i in range(line):
        if i > 0:
            str += eol
        percentage *= maxLen
        prog = int(percentage)
        percentage -= prog
        for j in range(prog):
            str += '=' if j != prog - 1 else '>'

    return str