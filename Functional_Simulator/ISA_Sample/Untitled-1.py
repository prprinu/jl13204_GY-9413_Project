import os

class DMEM(object):
    # Word addressible - each address contains 32 bits.
    def __init__(self, name, iodir, addressLen):
        self.name = name
        self.size = pow(2, addressLen)
        self.min_value  = -pow(2, 31)
        self.max_value  = pow(2, 31) - 1
        self.ipfilepath = os.path.abspath(os.path.join(iodir, name + ".txt"))
        self.opfilepath = os.path.abspath(os.path.join(iodir, name + "OP.txt"))
        self.data = []

        try:
            with open(self.ipfilepath, 'r') as ipf:
                self.data = [int(line.strip()) for line in ipf.readlines()]
            print(self.name, "- Data loaded from file:", self.ipfilepath)
            self.data.extend([0x0 for i in range(self.size - len(self.data))])
        except:
            print(self.name, "- ERROR: Couldn't open input file in path:", self.ipfilepath)

    def Read(self, idx): # Use this to read from DMEM.
        if idx < self.size:
            return self.data[idx]
        else:
            print("DMEM - ERROR: Invalid memory access at index:", idx, "with memory size:", self.size)

    def Write(self, idx, val): # Use this to write into DMEM.
        if idx < self.size:
            self.data[idx] = val
        else:
            print("DMEM - ERROR: Invalid memory write at index:", idx, "with memory size:", self.size)

    def dump(self):
        try:
            with open(self.opfilepath, 'w') as opf:
                lines = [str(data) + '\n' for data in self.data]
                opf.writelines(lines)
            print(self.name, "- Dumped data into output file in path:", self.opfilepath)
        except:
            print(self.name, "- ERROR: Couldn't open output file in path:", self.opfilepath)

# Testing the DMEM class
iodir = "C:\Users\jjlyu\Documents\ECE_GY_9413\Project\ISA_Sample"
dmem = DMEM("SDMEM", iodir, 13)  # Adjust the path and addressLen accordingly

# Perform some read and write operations
dmem.Write(0, 100)
value = dmem.Read(0)
print("Read value:", value)

# Dump the data to the output file
dmem.dump()
