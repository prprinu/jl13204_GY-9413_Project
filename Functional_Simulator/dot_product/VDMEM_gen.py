output_file = "VDMEM.txt"

with open(output_file, 'w') as file:
    for num in range(450):
        file.write(str(num) + '\n')