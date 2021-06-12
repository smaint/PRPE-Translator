
def fix_file(input):

    with open(input, 'r', encoding='utf8') as in_file, open(f"{input}.fixed", 'w', encoding='utf8') as out_file:
        lines = in_file.readlines()

        for line in lines:
            line = line.replace("’", "'")
            line = line.replace("‘", "'")

            out_file.write(line)

        in_file.close()
        out_file.close()

fix_file("../data/train.id.txt")