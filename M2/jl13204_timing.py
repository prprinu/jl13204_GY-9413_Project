import os
import argparse
import re

class Config(object):
    def __init__(self, iodir):
        self.filepath = os.path.abspath(os.path.join(iodir, "Config.txt"))
        self.parameters = {} # dictionary of parameter name: value as strings.

        try:
            with open(self.filepath, 'r') as conf:
                self.parameters = {line.split('=')[0].strip(): line.split('=')[1].split('#')[0].strip() for line in conf.readlines() if not (line.startswith('#') or line.strip() == '')}
            print("Config - Parameters loaded from file:", self.filepath)
            print("Config parameters:", self.parameters)
        except:
            print("Config - ERROR: Couldn't open file in path:", self.filepath)
            raise

class IMEM(object):
    def __init__(self, iodir):
        self.size = pow(2, 16) # Can hold a maximum of 2^16 instructions.
        self.filepath = os.path.abspath(os.path.join(iodir, "Code.asm"))
        self.instructions = []

        try:
            with open(self.filepath, 'r') as insf:
                self.instructions = [ins.strip() for ins in insf.readlines()]
            print("IMEM - Instructions loaded from file:", self.filepath)
            # print("IMEM - Instructions:", self.instructions)
        except:
            print("IMEM - ERROR: Couldn't open file in path:", self.filepath)

    def Read(self, idx): # Use this to read from IMEM.
        if idx < self.size:
            return self.instructions[idx]
        else:
            print("IMEM - ERROR: Invalid memory access at index: ", idx, " with memory size: ", self.size)

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
            # print(self.name, "- Data:", self.data)
            self.data.extend([0x0 for i in range(self.size - len(self.data))])
        except:
            print(self.name, "- ERROR: Couldn't open input file in path:", self.ipfilepath)

    def Read(self, idx): # Use this to read from DMEM.
        if idx < self.size:
            return self.data[idx]
        else:
            print("DMEM - ERROR: Invalid memory access at index: ", idx, " with memory size: ", self.size)
        

    def Write(self, idx, val): # Use this to write into DMEM.
        if idx < self.size:
            self.data[idx] = val
        else:
            print("DMEM - ERROR: Invalid memory write at index: ", idx, " with memory size: ", self.size)
        

    def dump(self):
        try:
            with open(self.opfilepath, 'w') as opf:
                lines = [str(data) + '\n' for data in self.data]
                opf.writelines(lines)
            print(self.name, "- Dumped data into output file in path:", self.opfilepath)
        except:
            print(self.name, "- ERROR: Couldn't open output file in path:", self.opfilepath)

class RegisterFile(object):
    def __init__(self, name, count, length=1, size=32):
        self.name = name
        self.reg_count = count
        self._vector_length = length  # Number of 32 bit words in a register.
        self.reg_bits = size
        self.min_value = -pow(2, self.reg_bits - 1)
        self.max_value = pow(2, self.reg_bits - 1) - 1
        self.registers = [[0x0 for e in range(self._vector_length)] for r in range(self.reg_count)]  # list of lists of integers

    @property
    def vector_length(self):
        return self._vector_length

    @vector_length.setter
    def vector_length(self, value):
        self._vector_length = value

    def Read(self, idx):
        if idx < 0 or idx >= self.reg_count:
            raise IndexError(f"Invalid memory access at index: {idx} with memory size: {self.reg_count}")
        return self.registers[idx]

    def Write(self, idx, val):
        if idx < 0 or idx >= self.reg_count:
            raise IndexError(f"Invalid memory write at index: {idx} with memory size: {self.reg_count}")
        if self.name == "SRF" and len(val) != 1:
            raise ValueError(f"Invalid value length for register with length {self._vector_length}")
        if self.name == "VRF" and len(val) != self._vector_length:
            raise ValueError(f"Invalid value length for register with length {self._vector_length}")
        self.registers[idx] = val

    def dump(self, iodir):
        opfilepath = os.path.abspath(os.path.join(iodir, self.name + ".txt"))
        try:
            with open(opfilepath, 'w') as opf:
                row_format = "{:<13}"*self._vector_length
                lines = [row_format.format(*[str(i) for i in range(self._vector_length)]) + "\n", '-'*(self._vector_length*13) + "\n"]
                lines += [row_format.format(*[str(val) for val in data]) + "\n" for data in self.registers]
                opf.writelines(lines)
            print(self.name, "- Dumped data into output file in path:", opfilepath)
        except:
            print(self.name, "- ERROR: Couldn't open output file in path:", opfilepath)

class MVDM(object):
    # Word addressible - each address contains 32 bits.
    def __init__(self, name, iodir, addressLen, num_banks):
        self.name = name
        self.size = pow(2, addressLen)
        self.num_banks = num_banks
        self.min_value  = -pow(2, 31)
        self.max_value  = pow(2, 31) - 1
        self.ipfilepath = os.path.abspath(os.path.join(iodir, name + ".txt"))
        self.opfilepath = os.path.abspath(os.path.join(iodir, name + "OP.txt"))
        
        # Initialize multi-bank data
        self.data = [[] for _ in range(num_banks)]

        try:
            with open(self.ipfilepath, 'r') as ipf:
                flat_data = [int(line.strip()) for line in ipf.readlines()]
            print(self.name, "- Data loaded from file:", self.ipfilepath)
            flat_data.extend([0x0 for i in range(self.size * self.num_banks - len(flat_data))])

            # Distribute data across banks
            for i in range(len(flat_data)):
                self.data[i % self.num_banks].append(flat_data[i])

        except:
            print(self.name, "- ERROR: Couldn't open input file in path:", self.ipfilepath)

    def Read(self, bank_idx, addr): # Use this to read from DMEM.
        if addr < self.size and bank_idx < self.num_banks:
            return self.data[bank_idx][addr]
        else:
            print("DMEM - ERROR: Invalid memory access at bank ", bank_idx, " and address: ", addr, " with memory size: ", self.size)
        

    def Write(self, bank_idx, addr, val): # Use this to write into DMEM.
        if addr < self.size and bank_idx < self.num_banks:
            self.data[bank_idx][addr] = val
        else:
            print("DMEM - ERROR: Invalid memory write at bank ", bank_idx, " and address: ", addr, " with memory size: ", self.size)
        

    def dump(self):
        try:
            with open(self.opfilepath, 'w') as opf:
                flat_data = [elem for bank_data in zip(*self.data) for elem in bank_data]
                lines = [str(data) + '\n' for data in flat_data]
                opf.writelines(lines)
            print(self.name, "- Dumped data into output file in path:", self.opfilepath)
        except:
            print(self.name, "- ERROR: Couldn't open output file in path:", self.opfilepath)

class DispatchQueue:
    def __init__(self, depth):
        self.queue = []
        self.depth = depth

    def is_full(self):
        return len(self.queue) == self.depth

    def is_empty(self):
        return len(self.queue) == 0

    def enqueue(self, instruction):
        if not self.is_full():
            self.queue.append(instruction)
            return True
        return False

    def dequeue(self):
        if not self.is_empty():
            return self.queue.pop(0)
        return None

    def peek(self):
        if not self.is_empty():
            return self.queue[0]
        return None

class Core():
    def __init__(self, imem, sdmem, vdmem, config):
        self.IMEM = imem
        self.SDMEM = sdmem
        self.VDMEM = vdmem

        self.RFs = {"SRF": RegisterFile("SRF", 8),
                    "VRF": RegisterFile("VRF", 8, 64)}
        
        # Your code here.
        self.VLR = [64]         # Vector Length Register, default to MVL
        self.RFs = {"SRF": RegisterFile("SRF", 8),
                    "VRF": RegisterFile("VRF", 8, self.VLR[0])}
        self.pc = 0
        self.cycle = 0
        self.VMR = [1 for x in range(64)] # Vector Mask Register, dedault to all ones
        # Dispatch Queue parameters
        self.dataQueueDepth = int(config.parameters['dataQueueDepth'])
        self.computeQueueDepth = int(config.parameters['computeQueueDepth'])

        # VDMEM LS parameters
        self.vdmNumBanks = int(config.parameters['vdmNumBanks'])
        self.vlsPipelineDepth = int(config.parameters['vlsPipelineDepth'])
        # Compute Pipeline parameters
        self.numLanes = int(config.parameters['numLanes'])
        self.pipelineDepthMul = int(config.parameters['pipelineDepthMul'])
        self.pipelineDepthAdd = int(config.parameters['pipelineDepthAdd'])
        self.pipelineDepthDiv = int(config.parameters['pipelineDepthDiv'])

        self.DQs = {"scalar": DispatchQueue(self.dataQueueDepth),
                    "compute": DispatchQueue(self.computeQueueDepth),
                    "vls": DispatchQueue(self.dataQueueDepth)}
        
        self.s_busy_board = [0 for i in range(8)]
        self.v_busy_board = [0 for i in range(8)]

    def run(self):
        while(True):
            # Fetch stage
            line = self.IMEM.Read(self.pc).strip()
            opcode = line.split(' ')[0]
            # branching instructions (__ takes six options: EQ, NE, GT, LT, GE, LE)
            if opcode in ['BEQ','BNE','BGT','BLT','BGE','BLE']:
                matches = re.findall(r"(?:SR)?(-?\d+)", line)
                rs1 = self.RFs["SRF"].Read(int(matches[0]))[0]
                rs2 = self.RFs["SRF"].Read(int(matches[1]))[0]
                Imm = int(line.split(' ')[3])
                max_value = pow(2, 20) - 1

                s = re.findall(r"SR(\d+)", line)
                for x in s:
                    if self.s_busy_board[int(x)] == 1:
                        opcode == 'STALL'
                        continue
                if opcode == 'STALL':
                    pass
                elif Imm > max_value or Imm < -max_value-1:
                    raise ValueError(f"Invalid immediate value {Imm} for branch instruction")
                elif (opcode == 'BEQ' and rs1 == rs2) or (opcode == 'BNE' and rs1 != rs2) or (opcode == 'BGT' and rs1 > rs2) or (opcode == 'BLT' and rs1 < rs2) or (opcode == 'BGE' and rs1 >= rs2) or (opcode == 'BLE' and rs1 <= rs2):
                    self.pc += Imm 
                self.cycle += 1

            elif opcode == 'HALT':
                ##raise SystemExit("HALT encountered, exiting...")
                print("HALT encountered, exiting...")
                self.cycle += 1
                print("Total cycles: ", self.cycle)
                break

            else:
                # Decode stage
                s = re.findall(r"SR(\d+)", line)
                for x in s:
                    if int(x) >= 8 or int(x) < 0:
                        raise ValueError(f"Invalid register number {x} for SRF")
                    elif self.s_busy_board[int(x)] == 1:
                        opcode == 'STALL'
                    else: self.s_busy_board[int(x)] = 1
                v = re.findall(r"VR(\d+)", line)
                for x in v:
                    if int(x) >= 8 or int(x) < 0:
                        raise ValueError(f"Invalid register number {x} for SRF")
                    elif self.v_busy_board[int(x)] == 1:
                        opcode == 'STALL'
                    else: self.v_busy_board[int(x)] = 1
                self.cycle += 1

                # Dispatch stage (queues)
                if opcode in ['MTCL','CVM','LV','SV','LVWS','SVWS','LVI','SVI']:
                    if self.DQs["vls"].is_full():
                        self.pc -= 1
                    else: self.DQs["vls"].enqueue(line)
                elif opcode in ['ADDVV','SUBVV','MULVV','DIVVV','ADDVS','SUBVS','MULVS','DIVVS','SEQVV','SNEVV','SGTVV','SLTVV','SGEVV','SLEVV','SEQVS','SNEVS','SGTVS','SLTVS','SGEVS','SLEVS',]:
                    if self.DQs["compute"].is_full():
                        self.pc -= 1
                    else: self.DQs["compute"].enqueue(line)
                elif opcode in ['ADD','SUB','AND','OR','XOR','SLL','SRL','SRA','LS','SS','POP','MFCL']:
                    if self.DQs["scalar"].is_full():
                        self.pc -= 1
                    else: self.DQs["scalar"].enqueue(line)
                self.cycle += 1

            # Execute stage
            self.cycle += 1
            # Vector L/S
            vls = self.DQs["vls"].dequeue()
            if vls is not None:
                line = vls.strip()
                opcode = line.split(' ')[0]
                self.pc += 1

                # Clear busy board
                v = re.findall(r"SR(\d+)", line)
                for x in v:
                    self.v_busy_board[int(x)] = 0
                s = re.findall(r"SR(\d+)", line)
                for x in s:
                    self.s_busy_board[int(x)] = 0

                # Vector Length Register Operations
                if opcode == 'MTCL':
                    matches = re.findall("\d+", line)
                    self.VLR = self.RFs["SRF"].Read(int(matches[0]))
                    self.RFs["VRF"].vector_length = self.VLR[0]

                # Vector Mask Register Operations   
                elif opcode == 'CVM':
                    self.VMR = [1 for x in range(self.VLR[0])] # Clear

                # Memory Access Operations    
                if opcode == 'LV':
                    vrd = [0 for x in range(self.VLR[0])]
                    matches = re.findall("\d+", line) 
                    addr = self.RFs["SRF"].Read(int(matches[1]))
                    for n in range(self.VLR[0]):  
                        if self.VMR[n] == 1:                         
                            vrd[n] = self.VDMEM.Read(addr[0]+n)
                    self.RFs["VRF"].Write(int(matches[0]),vrd)
                if opcode == 'SV':
                    matches = re.findall("\d+", line)
                    vrs1 = self.RFs["VRF"].Read(int(matches[0]))
                    addr = self.RFs["SRF"].Read(int(matches[1]))
                    for n in range(self.VLR[0]):  
                        if self.VMR[n] == 1: self.VDMEM.Write(addr[0]+n,vrs1[n])
                if opcode == 'LVWS':
                    matches = re.findall("\d+", line) 
                    addr = self.RFs["SRF"].Read(int(matches[1]))
                    stride = self.RFs["SRF"].Read(int(matches[2])) #stride
                    vrd = [0 for x in range(self.VLR[0])]
                    for n in range(self.VLR[0]):  
                        if self.VMR[n] ==1: vrd[n] = self.VDMEM.Read(addr[0]+stride[0]*n)
                    self.RFs["VRF"].Write(int(matches[0]),vrd)
                if opcode == 'SVWS':
                    matches = re.findall("\d+", line)
                    vrs1 = self.RFs["VRF"].Read(int(matches[0])) #VR
                    addr = self.RFs["SRF"].Read(int(matches[1])) #SR          
                    stride = self.RFs["SRF"].Read(int(matches[2])) #stride
                    for n in range(self.VLR[0]):  
                        if self.VMR[n] ==1: self.VDMEM.Write(addr[0]+stride[0]*n,vrs1[n])
                if opcode == 'LVI':
                    matches = re.findall("\d+", line)
                    addr = self.RFs["SRF"].Read(int(matches[1]))
                    vrs2 = self.RFs["VRF"].Read(int(matches[2]))
                    vrd = [0 for x in range(self.VLR[0])]
                    for n in range(self.VLR[0]):  
                        if self.VMR[n] == 1: vrd[n] = self.VDMEM.Read(addr[0]+vrs2[n])                       
                    self.RFs["VRF"].Write(int(matches[0]),vrd)
                if opcode == 'SVI':
                    matches = re.findall("\d+", line)
                    addr = self.RFs["SRF"].Read(int(matches[1]))
                    vrs1 = self.RFs["VRF"].Read(int(matches[0]))
                    vrs2 = self.RFs["VRF"].Read(int(matches[2]))
                    for n in range(self.VLR[0]):  
                        if self.VMR[n] == 1: self.VDMEM.Write(addr[0]+vrs2[n],vrs1[n])
            
            # Vector computation
            compute = self.DQs["compute"].dequeue()
            if compute is not None:
                line = compute.strip()
                opcode = line.split(' ')[0]
                self.pc += 1

                # Clear busy board
                v = re.findall(r"SR(\d+)", line)
                for x in v:
                    self.v_busy_board[int(x)] = 0
                s = re.findall(r"SR(\d+)", line)
                for x in s:
                    self.s_busy_board[int(x)] = 0

                if opcode in ['ADDVV','SUBVV','MULVV','DIVVV']:
                    matches = re.findall("\d+", line) # find the register number in integer
                    vrs1 = self.RFs["VRF"].Read(int(matches[1]))
                    vrs2 = self.RFs["VRF"].Read(int(matches[2]))
                    vrd = [0 for x in range(self.VLR[0])]
                    max_value = pow(2, 31) - 1
                    if opcode == 'ADDVV':
                        for x in range (self.VLR[0]):
                            tmp = vrs1[x] + vrs2[x]                       
                            if tmp > max_value: tmp = max_value                        
                            elif tmp < -max_value-1: tmp = -max_value-1
                            elif self.VMR[x] == 1: vrd[x] = tmp                 
                    elif opcode == 'SUBVV':
                        for x in range (self.VLR[0]):                        
                            tmp = vrs1[x] - vrs2[x]                       
                            if tmp > max_value: tmp = max_value                        
                            elif tmp < -max_value-1: tmp = -max_value-1
                            elif self.VMR[x] == 1: vrd[x] = tmp
                    elif opcode == 'MULVV': 
                        for x in range (self.VLR[0]):
                            tmp = vrs1[x] * vrs2[x]                       
                            if tmp > max_value: tmp = max_value                        
                            elif tmp < -max_value-1: tmp = -max_value-1
                            elif self.VMR[x] == 1: vrd[x] = tmp
                    elif opcode == 'DIVVV':     
                        for x in range (self.VLR[0]): 
                            if vrs2[x] == 0:
                                ##raise ZeroDivisionError("Division by zero encountered")
                                pass                         
                            elif self.VMR[x] == 1: vrd[x] = vrs1[x] // vrs2[x]
                    self.RFs["VRF"].Write(int(matches[0]),vrd)
            
                elif opcode in ['ADDVS','SUBVS','MULVS','DIVVS']:
                    matches = re.findall("\d+", line) # find the register number in integer
                    vrs1 = self.RFs["VRF"].Read(int(matches[1]))
                    rs2 = self.RFs["SRF"].Read(int(matches[2]))
                    vrd = [0 for x in range(self.VLR[0])]
                    max_value = pow(2, 31) - 1
                    if opcode == 'ADDVS':
                        for x in range(self.VLR[0]):
                            tmp = vrs1[x] + rs2[0]                       
                            if tmp > max_value: tmp = max_value                        
                            elif tmp < -max_value-1: tmp = -max_value-1
                            elif self.VMR[x] == 1: vrd[x] = tmp
                    elif opcode == 'SUBVS':
                        for x in range(self.VLR[0]):
                            tmp = vrs1[x] - rs2[0]                       
                            if tmp > max_value: tmp = max_value                        
                            elif tmp < -max_value-1: tmp = -max_value-1
                            elif self.VMR[x] == 1: vrd[x] = tmp
                    elif opcode == 'MULVS':
                        for x in range(self.VLR[0]):
                            tmp = vrs1[x] * rs2[0]                       
                            if tmp > max_value: tmp = max_value                        
                            elif tmp < -max_value-1: tmp = -max_value-1
                            elif self.VMR[x] == 1: vrd[x] = tmp
                    elif opcode == 'DIVVS':                   
                        if rs2[0] == 0:
                            ##raise ZeroDivisionError("Division by zero encountered")
                            pass                         
                        elif self.VMR[x] == 1: 
                            for x in range(self.VLR[0]):
                                vrd[x] = vrs1[x] // rs2[0]
                    self.RFs["VRF"].Write(int(matches[0]),vrd)

                elif opcode in ['SEQVV','SNEVV','SGTVV','SLTVV','SGEVV','SLEVV']:
                    matches = re.findall("\d+", line) # find the register number in integer
                    vrs1 = self.RFs["VRF"].Read(int(matches[0]))
                    vrs2 = self.RFs["VRF"].Read(int(matches[1]))
                    if opcode == 'SEQVV':
                        for n in range(self.VLR[0]):
                            if vrs1[n] == vrs2[n]:                                      
                                self.VMR[n] = 1    # set the nth bit to 1
                    elif opcode == 'SNEVV':           
                        for n in range(self.VLR[0]):
                            if vrs1[n] != vrs2[n]:                                      
                                self.VMR[n] = 1    
                    elif opcode == 'SGTVV':
                        for n in range(self.VLR[0]):
                            if vrs1[n] > vrs2[n]:                                      
                                self.VMR[n] = 1    
                    elif opcode == 'SLTVV':
                        for n in range(self.VLR[0]):
                            if vrs1[n] < vrs2[n]:                                      
                                self.VMR[n] = 1    
                    elif opcode == 'SGEVV':
                        for n in range(self.VLR[0]):
                            if vrs1[n] >= vrs2[n]:                                      
                                self.VMR[n] = 1    
                    elif opcode == 'SLEVV':
                        for n in range(self.VLR[0]):
                            if vrs1[n] <= vrs2[n]:                                     
                                self.VMR[n] = 1    

                if opcode in ['SEQVS','SNEVS','SGTVS','SLTVS','SGEVS','SLEVS']:
                    matches = re.findall("\d+", line) # find the register number in integer
                    vrs1 = self.RFs["VRF"].Read(int(matches[0]))
                    rs2 = self.RFs["SRF"].Read(int(matches[1]))[0]
                    if opcode == 'SEQVS':
                        for n in range(self.VLR[0]):
                            if vrs1[n] == rs2:                                      
                                self.VMR[n] = 1    
                    elif opcode == 'SNEVS':           
                        for n in range(self.VLR[0]):
                            if vrs1[n] != rs2:                                      
                                self.VMR[n] = 1    
                    elif opcode == 'SGTVS':
                        for n in range(self.VLR[0]):
                            if vrs1[n] > rs2:                                     
                                self.VMR[n] = 1    
                    elif opcode == 'SLTVS':
                        for n in range(self.VLR[0]):
                            if vrs1[n] < rs2:                                   
                                self.VMR[n] = 1    
                    elif opcode == 'SGEVS':
                        for n in range(self.VLR[0]):
                            if vrs1[n] >= rs2:                                   
                                self.VMR[n] = 1    
                    elif opcode == 'SLEVS':
                        for n in range(self.VLR[0]):
                            if vrs1[n] <= rs2:                                   
                                self.VMR[n] = 1
                    
            
            # Scalar Operation
            scalar = self.DQs["scalar"].dequeue()
            if scalar is not None:
                line = scalar.strip()
                opcode = line.split(' ')[0]
                self.pc += 1

                # Clear busy board
                s = re.findall(r"SR(\d+)", line)
                for x in s:
                    self.s_busy_board[int(x)] = 0
                    
                if opcode == 'POP':
                    matches = re.findall("\d+", line)
                    count = [self.VMR.count(1)]    # count 1
                    self.RFs["SRF"].Write(int(matches[0]),count)

                elif opcode == 'MFCL':
                    matches = re.findall("\d+", line)
                    self.RFs["SRF"].Write(int(matches[0]),self.VLR)
                
                elif opcode == 'LS':
                    Imm = int(line.split(' ')[3])
                    matches = re.findall("\d+", line)
                    addr = self.RFs["SRF"].Read(int(matches[1]))
                    self.RFs["SRF"].Write(int(matches[0]),[self.SDMEM.Read(addr[0]+Imm)])
                elif opcode == 'SS':
                    Imm = int(line.split(' ')[3])
                    matches = re.findall("\d+", line)
                    addr = self.RFs["SRF"].Read(int(matches[1]))
                    rs2 = self.RFs["SRF"].Read(int(matches[0]))
                    self.SDMEM.Write(addr[0]+Imm,rs2[0])
                
                # Scalar Operations
                elif opcode in ['ADD','SUB','AND','OR','XOR','SLL','SRL','SRA']:
                    matches = re.findall("\d+", line)
                    rs1 = self.RFs["SRF"].Read(int(matches[1]))[0]
                    rs2 = self.RFs["SRF"].Read(int(matches[2]))[0]
                    rd = 0     
                    max_value = pow(2, 31) - 1          
                    if opcode == 'ADD':
                        rd = rs1+rs2
                        if rd > max_value: rd = max_value                        
                        elif rd < -max_value-1: rd = -max_value-1
                    elif opcode == 'SUB':
                        rd = rs1-rs2
                        if rd > max_value: rd = max_value                        
                        elif rd < -max_value-1: rd = -max_value-1
                    elif opcode == 'AND':
                        rd = rs1&rs2
                    elif opcode == 'OR':
                        rd = rs1|rs2
                    elif opcode == 'XOR':
                        rd = rs1^rs2
                    elif opcode == 'SLL':
                        rd = (rs1<<rs2) & 0xffffffff # 32-bit mask
                    elif opcode == 'SRL':
                        rd = rs1>>rs2
                    elif opcode == 'SRA':
                        sign = rs1 & (1 << 31)
                        # perform SRA by shifting right and preserving the sign bit
                        rd = (rs1 >> rs2) | sign
                    self.RFs["SRF"].Write(int(matches[0]),[rd])
                

    def dumpregs(self, iodir):
        for rf in self.RFs.values():
            rf.dump(iodir)

if __name__ == "__main__":
    #parse arguments for input file location
    parser = argparse.ArgumentParser(description='Vector Core Functional Simulator')
    parser.add_argument('--iodir', default="", type=str, help='Path to the folder containing the input files - instructions and data.')
    args = parser.parse_args()

    iodir = os.path.abspath(args.iodir)
    print("IO Directory:", iodir)

    # Parse Config
    config = Config(iodir)

    # Parse IMEM
    imem = IMEM(iodir)  
    # Parse SMEM
    sdmem = DMEM("SDMEM", iodir, 13) # 32 KB is 2^15 bytes = 2^13 K 32-bit words.
    # Parse VMEM
    vdmem = MVDM("VDMEM", iodir, 17, config.parameters("vdmNumBanks")) # 512 KB is 2^19 bytes = 2^17 K 32-bit words. 

    # Create Vector Core
    vcore = Core(imem, sdmem, vdmem, config)

    # Run Core
    vcore.run()   
    vcore.dumpregs(iodir)

    sdmem.dump()
    vdmem.dump()

    # THE END