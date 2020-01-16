

def read_in_indices_from_txt(filename):
    indices = []
    with open(filename) as f:
        for line in f:
            indices.append(int(line.rstrip('\n')))
    return indices
