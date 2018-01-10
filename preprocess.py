def convert_file(filename):
  with open(filename) as inputfile:
    with open(filename + '.conv.xyz', 'w') as outputfile:
      molecule = list(filter(lambda s: "#" not in s, inputfile.readlines()))
      outputfile.write('{}\n'.format(str(len(molecule) - 3)))
      for line in molecule[3:]:
        _, x, y, z, a = line.split(' ')
        outputfile.write('{} {} {} {}\n'.format(a.strip(), x.strip(), y.strip(), z.strip()))

convert_file('train/1/geometry.xyz')