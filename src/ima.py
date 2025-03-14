# Defines a configurable IMA module with its methods

# add the folder location for include files
import sys, json
import time

from line_profiler import profile

# import dependancy files
import numpy as np
import math
import include.config as cfg
from include.data_config import datacfg
#import include.configTest as cfg
import include.constants as param
import include.constants_digital as digi_param
import src.ima_modules as imod

from data_convert import *

#phy2log_ratio = cfg.phy2log_ratio # ratio of physical to logical xbar / will not be used any more. replaced by datacfg.ReRAM_xbar_num
# datamem_off is the start of address space of datamemory
datamem_off = cfg.datamem_off # each matrix has 6 memory spaces (1 for f/b, 2 for d)
my_xbar_count = 0

# This is a debug feature which allows tracking one particular value
tracking = False # set to True to track a particular value
track_mat_id = 0 # matrix id to track
track_type = 'f' # type of xbar to track (f/b/d)
track_addr = 11 # address in xbar to track(0-127)

class ima (object):

    instances_created = 0

    #######################################################
    ### Instantiate different modules
    #######################################################
    
    def __init__ (self):

        # Assign a ima_id for identification purpose in debug trace
        self.ima_id = ima.instances_created
        ima.instances_created += 1

        ######################################################################
        ## Parametrically instantiate different physical IMA hardware modules
        ######################################################################

        # Instantiate xbar, xbar_inMem, xbar_outMem -components store states specific to a xbar
        self.matrix_list = [] # list of dicts of mvmu(s)
        self.xb_inMem_list = [] # list of dicts of xbar input memory
        self.xb_outMem_list = [] # list of dicts of xbar output memory

        for i in range(cfg.num_matrix):
            # each matrix represents three mvmus - 1 mvmu for fw, 1 mvmu for bw, 1 mvmu (2X width) for delta
            temp_xbar_dict = {'f':[], 'b':[], 'd':[]}
            temp_inMem_dict = {'f':[], 'b':[], 'd':[]}
            temp_outMem_dict = {'f':[], 'b':[], 'd':[]}

            for key in temp_xbar_dict:
                #phy2log_ratio = cfg.data_width/cfg.xbar_bits # ratio of physical to logical xbars
                #numXbar_temp = (2*phy2log_ratio) if (key == 'd') else (phy2log_ratio)

                # assign xbars to the dict elements
                temp_list_xbar = []
                for resolution in datacfg.storage_config:
                    if resolution != 's':
                        resolution = int(resolution)
                        if (key != 'd'):
                            temp_xbar = imod.xbar (cfg.xbar_size, resolution)
                            temp_list_xbar.append (temp_xbar)
                        else:
                            temp_xbar = imod.xbar_op (cfg.xbar_size, resolution)
                            # For delta we have 2X width
                            temp_list_xbar.append (temp_xbar)
                            temp_list_xbar.append (temp_xbar)
                    else:
                        # TODO add SRAM instantiate
                        True
                temp_xbar_dict[key] = temp_list_xbar
                # assign input memory to mvmu
                temp_inMem_dict[key] = imod.xb_inMem (cfg.xbar_size)

                # assign output memory to mvmu
                temp_outMem_dict[key] = imod.xb_outMem (cfg.xbar_size)

            self.matrix_list.append(temp_xbar_dict)
            self.xb_inMem_list.append(temp_inMem_dict)
            self.xb_outMem_list.append(temp_outMem_dict)


        # Instantiate DACs
        self.dacArray_list = [] # list of dicts
        # each matrix will have mutiple dac_arrays for each of its mvmu (f,b,d)
        for i in range(cfg.num_matrix):
            temp_dict = {'f':[], 'b':[], 'd_r':[], 'd_c':[]} # separate dac_array for delta xbar row and columns
            for key in temp_dict:
                if (key in ['f', 'b', 'd_r']):
                    temp_dacArray = imod.dac_array (cfg.xbar_size, cfg.dac_res)
                else:
                    # 2-bit (=xbar_bits) are fed to columns of crossbar)
                    # TODO needs to fix, since different res is needed.
                    temp_dacArray = imod.dac_array (cfg.xbar_size, 2*cfg.dac_res)
                temp_dict[key] = temp_dacArray
            self.dacArray_list.append(temp_dict)

        # Instatiate ADCs
        # No ADC needed for delta xbar, so key is f,b
        # When using normal ADC, list will be in form of [matrix_id][key][xbar_id][adc_id][pos/neg]
        # When using differential ADC, list will be in form of [matrix_id][key][xbar_id][adc_id]
        self.adc_list = []
        self.adc_type = cfg.adc_type
        adc_per_xbar = cfg.xbar_size // cfg.num_column_per_adc
        for mat_id in range(cfg.num_matrix):
            self.adc_list.append({})
            for key in ['f', 'b']:
                self.adc_list[mat_id][key] = []
                for xbar_id in range(datacfg.ReRAM_xbar_num):
                    self.adc_list[mat_id][key].append([])
                    for adc_id in range(adc_per_xbar):
                        if cfg.adc_type == 'normal': # two normal ADCs will be needed here, one for positive and one for negative
                            self.adc_list[mat_id][key][xbar_id].append({})
                            for pos_neg in ['pos', 'neg']:
                                temp_adc = imod.adc (cfg.adc_res)
                                self.adc_list[mat_id][key][xbar_id][adc_id][pos_neg] = temp_adc
                        elif cfg.adc_type == 'differential':
                            temp_adc = imod.differential_adc (cfg.adc_res)
                            self.adc_list[mat_id][key][xbar_id].append(temp_adc)

        # Instantiate sample and hold
        self.snh_list_pos = []
        self.snh_list_neg = []
        for i in range (2 * cfg.num_matrix * datacfg.ReRAM_xbar_num):
            temp_snh_pos = imod.sampleNhold (cfg.xbar_size)
            temp_snh_neg = imod.sampleNhold (cfg.xbar_size)
            self.snh_list_pos.append(temp_snh_pos)
            self.snh_list_neg.append(temp_snh_neg)

        # Instatiate MUX
        # MUX list will be in form of [matrix_id][key][xbar_id][mux_id][pos/neg]
        self.mux_list = []
        mux_per_xbar = cfg.xbar_size // cfg.num_column_per_adc
        for mat_id in range(cfg.num_matrix):
            self.mux_list.append({})
            for key in ['f', 'b']:
                self.mux_list[mat_id][key] = []
                for xbar_id in range(datacfg.ReRAM_xbar_num):
                    self.mux_list[mat_id][key].append([])
                    for mux_id in range(mux_per_xbar):
                        self.mux_list[mat_id][key][xbar_id].append({})
                        for pos_neg in ['pos', 'neg']:
                            temp_mux = imod.mux (cfg.num_column_per_adc)
                            self.mux_list[mat_id][key][xbar_id][mux_id][pos_neg] = temp_mux

        # Instantiate ALUs
        self.alu_list = []
        for i in range(cfg.num_ALU):
            temp_alu = imod.alu ()
            self.alu_list.append(temp_alu)

        # Instantiate integger ALU
        self.alu_int = imod.alu_int ()

        # Instantiate  data memory (stores data)
        self.dataMem = imod.memory (cfg.dataMem_size, cfg.datamem_off)

        # Instantiate instruction memory (stores instruction)
        self.instrnMem = imod.instrn_memory (cfg.instrnMem_size)

        # Instantiate the memory interface (interface to edram controller)
        self.mem_interface = imod.mem_interface ()

        #############################################################################################################
        ## Define virtual (currently for software emulation purpose (doesn't have a corresponding hardware currenty)
        #############################################################################################################

        # Define stage-wise pipeline registers (f - before fetch, fd -fetch_decode, de - decode_execute)
        self.pc = 0 # holds the next program counter value

        self.fd_instrn = param.dummy_instrn

        self.de_instrn = param.dummy_instrn # For Debug Only

        self.de_opcode = param.dummy_instrn['opcode']
        self.de_aluop = param.dummy_instrn['aluop']
        self.de_d1 = param.dummy_instrn['d1'] # target register addr for alu/alui/ld
        self.de_imm = param.dummy_instrn['imm'] # imm value for alui
        self.de_xb_nma = param.dummy_instrn['xb_nma'] # nma value for xbar execution

        self.de_r1 = 0 # operand addr read from r1 address
        self.de_r2 = 0 # operand addr read from r2 address
        self.de_val1 = 0 # operand value
        self.de_val2 = 0 # opearnd value
        self.de_vec = 1 # vector width

        self.ex_vec_count = 0

        ########################################################
        ## Define book-keeping variables for pipeline execution
        ########################################################
        self.num_stage = len (param.stage_list)

        # Tells when EDRAM access for ld instruction is done
        self.ldAccess_done = 0

        # Define the book-keeping variables - stage-specific
        self.stage_empty = [0] * self.num_stage
        self.stage_cycle = [0] * self.num_stage
        self.stage_latency = [0] * self.num_stage # tells how many cycles will the current method running in a stage will require
        self.stage_done = [0] * self.num_stage

        # Define global pipeline variables
        self.debug = 0

        # Define a halt signal
        self.halt = 0

        # Define a counter to compute leak_energy
        self.cycle_count = 0 # (power-gated imas - before they start and after they halt)

    # Function to read the content of a matrix (from physical xbars to logical xbar)
    def get_matrix (self, mat_id, key):
        matrix = np.zeros((cfg.xbar_size, cfg.xbar_size))
        for k in range (cfg.xbar_size):
            for l in range (cfg.xbar_size):
                # read wt slices from delta xbar to compose a new weight
                wt_new = 0.0
                for m in datacfg.ReRAM_xbar_num:
                    if key in ['f', 'd']:
                        wt_new += self.matrix_list[mat_id][key][m].read(k,l) * (2 ** (datacfg.stored_bit[m] - datacfg.frac_bits)) # left shift
                    else:
                        wt_new += self.matrix_list[mat_id][key][2 * m].read(k,l) * (2 ** (2 * datacfg.stored_bit[m] - datacfg.frac_bits)) # left shift
                        wt_new += self.matrix_list[mat_id][key][2 * m + 1].read(k,l) * (2 ** (2 * datacfg.stored_bit[m] + datacfg.storage_config[m] - datacfg.frac_bits)) # left shift
                matrix[k][l] = wt_new
        return matrix


    ############################################################
    ### Define what a pipeline stage does for each instruction
    ############################################################
    # Increment stage cycles but update pipeline registers at end only when update_ready flag is set

    # "Fetch" stage (common to all instructions)
    def fetch (self, update_ready, fid):
        sId = 0 # sId - stageId

        # Define what to do in fetch
        def do_fetch (self):
            # commmon to all instructions
            self.fd_instrn = self.instrnMem.read (self.pc) # update pipeline register (fetch/decode)

            # A blank instruction signifies program end
            if (self.fd_instrn != ''):
                self.stage_empty[sId+1] = 0
                self.stage_done[sId+1] = 0

                self.pc = self.pc + 1 # update pipeline register before fetch stage
                # self.stage_empty[sId] = 1


        # Describe the functionality on a cycle basis
        # Start a fetch stage - if fetch stage is empty and succedding stage is done (update_ready)
        # Succeding stages back-propagate update_ready when they are done
        # For all other stages (except fetch) - start when stage is non-empty

        # State machine (lil different than other stages)
        # Describe the functionality on a cycle basis
        if (self.stage_empty[sId] != 1):
            # First cycle - update the target latency
            if (self.stage_cycle[sId] == 0):
                self.stage_latency[sId] = self.instrnMem.getLatency()

                # Check if first = last cycle
                if (self.stage_latency[sId] == 1 and update_ready and (not self.halt)):
                    do_fetch (self)
                    self.stage_done[sId] = 1
                    self.stage_cycle[sId] = 0
                    #self.stage_empty[sId] = 1
                else:
                    self.stage_cycle[sId] = self.stage_cycle[sId] + 1

            # Last cycle - update pipeline registers & done flag
            elif (self.stage_cycle[sId] >= self.stage_latency[sId]-1 and update_ready and (not self.halt)):
                do_fetch (self)
                self.stage_done[sId] = 1
                self.stage_cycle[sId] = 0
                #self.stage_empty[sId] = 1

            # For all other cycles
            else:
                self.stage_cycle[sId] = self.stage_cycle[sId] + 1


    # "Decode" stage - Reads operands (if needed) and puts into the specific data structures
    def decode (self, update_ready, fid):
        sId = 1

        # Define what to do in decode (done for conciseness)
        def do_decode (self, dec_op):
            # common to all instructions
            self.de_opcode = dec_op
            self.stage_empty[sId+1] = 0
            self.stage_done[sId+1] = 0

            self.de_instrn = self.fd_instrn

            # instruction specific (for eg: ld_dec - load's decode stage)
            if (dec_op == 'ld'):
                assert (self.fd_instrn['r1'] >= datamem_off), 'load address for tile memory comes from data memory'
                self.de_r1 = bin2int(self.dataMem.read(self.fd_instrn['r1']), cfg.addr_width) # absolute mem addr
                assert (self.de_r1 >=0) # mem addr for load should be non negative
                self.de_d1 = self.fd_instrn['d1']
                self.de_r2 = self.fd_instrn['imm'] # used for incrementing/decrementing counter for edram entries
                self.de_vec = self.fd_instrn['vec']

            elif (dec_op == 'cp'):
                self.de_d1 = self.fd_instrn['d1'] # reg addr
                self.de_r1 = self.fd_instrn['r1'] # reg addr
                self.de_r2 = self.fd_instrn['r2'] # reg addr
                self.de_vec = self.fd_instrn['vec']
                # source value will be read in execute stage

            elif (dec_op == 'st'):
                assert (self.fd_instrn['d1'] >= datamem_off), 'store address for tile memory comes from data memory'
                self.de_d1 = bin2int(self.dataMem.read(self.fd_instrn['d1']), cfg.addr_width) #absolute mem addr
                assert (self.de_d1 >=0) # mem addr for store should be non negative
                self.de_r1 = self.fd_instrn['r1'] # reg addr
                self.de_vec = self.fd_instrn['vec']
                # source value will be read in execute stage
                # NEW - added store counter (comes from r2 and stored in val2)
                self.de_val1 = self.fd_instrn['r2']
                self.de_r2 = self.fd_instrn['imm'] # used for incrementing/decrementing counter for edram entries

            elif (dec_op == 'set'):
                self.de_d1 = self.fd_instrn['d1'] # addr for rf
                self.de_val1 = self.fd_instrn['imm'] #absolute value (shift)
                self.de_vec = self.fd_instrn['vec']
                self.de_r1 = self.fd_instrn['r1'] # is address or data? address = 1, data = 0

            elif (dec_op == 'alu'):
                self.de_aluop = self.fd_instrn['aluop']
                self.de_d1 = self.fd_instrn['d1'] # addr for rf
                self.de_r1 = self.fd_instrn['r1'] #addr for rf
                self.de_r2 = self.fd_instrn['r2'] #addr for rf
                self.de_val1 = self.fd_instrn['imm'] #absolute value (shift)
                self.de_vec = self.fd_instrn['vec']
                # source values (operands) will be read in execute stage

            elif (dec_op == 'alui'):
                self.de_aluop = self.fd_instrn['aluop']
                self.de_d1 = self.fd_instrn['d1'] # addr for rf
                self.de_r1 = self.fd_instrn['r1'] #addr for rf
                self.de_val1 = self.fd_instrn['imm'] #absolute value (shift)
                assert (len(self.de_val1) == cfg.num_bits), 'imm values must be datawidth bit strings'
                self.de_vec = self.fd_instrn['vec']
                # source value will be read in execute stage

            elif (dec_op == 'mvm'):
                xb_nma = self.fd_instrn['xb_nma']
                assert (len(xb_nma) == cfg.num_matrix), 'unsupported xbar configuration'
                self.de_xb_nma = xb_nma
                # adding a value for stride at the end of mvm processing (for input sharing across strides)
                self.de_val1 = self.fd_instrn['r1']
                self.de_val2 = self.fd_instrn['r2']

            elif (dec_op == 'crs'):
                xb_nma = self.fd_instrn['xb_nma']
                assert (len(xb_nma) == cfg.num_matrix), 'unsupported xbar configuration'
                self.de_xb_nma = xb_nma

            elif (dec_op == 'beq'):
                self.de_aluop = 'eq_chk' # equality check with integer ALU
                assert (self.fd_instrn['r1'] >= datamem_off), 'operand1 for beq comes from data memory'
                assert (self.fd_instrn['r2'] >= datamem_off), 'operand2 for beq comes from data memory'
                self.de_val1 = self.dataMem.read(self.fd_instrn['r1'])
                self.de_val2 = self.dataMem.read(self.fd_instrn['r2'])

            elif (dec_op == 'alu_int'):
                self.de_aluop = self.fd_instrn['aluop']
                self.de_d1 = self.fd_instrn['d1'] # addr for rf
                assert (self.fd_instrn['r1'] >= datamem_off), 'operand1 for alu_int comes from data memory'
                assert (self.fd_instrn['r2'] >= datamem_off), 'operand2 for alu_int comes from data memory'
                self.de_val1 = self.dataMem.read(self.fd_instrn['r1'])
                self.de_val2 = self.dataMem.read(self.fd_instrn['r2'])

            # do nothing for halt/jmp in decode (just propagate to ex when applicable)


        # State machine runs only if the stage is non-empty
        # Describe the functionality on a cycle basis
        # Decode stage has a fixed latency always - datamem read latency
        if (self.stage_empty[sId] != 1):
            # First cycle - update the target latency
            if (self.stage_cycle[sId] == 0):
                # Check for assertion pass
                dec_op = self.fd_instrn['opcode']
                assert (dec_op in param.op_list), 'unsupported opcode'

                self.stage_latency[sId] = self.dataMem.getLatency()

                # Check if first = last cycle
                if (self.stage_latency[sId] == 1 and update_ready and (not self.halt)):
                    do_decode (self, dec_op)
                    self.stage_done[sId] = 1
                    self.stage_cycle[sId] = 0
                    self.stage_empty[sId] = 1
                else:
                    self.stage_cycle[sId] = self.stage_cycle[sId] + 1

            # Last cycle - update pipeline registers (if ??) & done flag
            elif (self.stage_cycle[sId] >= self.stage_latency[sId]-1 and update_ready and (not self.halt)):
                dec_op = self.fd_instrn['opcode']
                do_decode (self, dec_op)
                self.stage_done[sId] = 1
                self.stage_cycle[sId] = 0
                self.stage_empty[sId] = 1

            # For all other cycles (non-first, non-last, non-update ready)
            else:
                self.stage_cycle[sId] = self.stage_cycle[sId] + 1


    # Execute stage - compute and store back to registers
    def execute (self, update_ready, fid):
        sId = 2

        # define some common functions use dto address xbar memory spaces
        # xbar memory spaces are addressed as num_mvmu, f,b/d, i/o order
        # find [num_matrix, xbar_type, mem_addr, xbar_addr]
        def getXbarAddr (data_addr):
            
            if (cfg.training):
                # find i or o
                if (data_addr < cfg.num_matrix*3*cfg.xbar_size):
                    mem_addr = 0
                else:
                    # NOTE why? thinking if it would be better to have mem_addr = cfg.xbar_size Maybe as a mark it doesn't matter
                    mem_addr = cfg.xbar_size

                # find xbar_addr
                xbar_addr = data_addr % cfg.xbar_size

                # find matrix_addr
                num_matrix = (data_addr // (3*cfg.xbar_size)) % cfg.num_matrix

                # find xbar_type
                temp_val = (data_addr % (cfg.num_matrix*3*cfg.xbar_size))
                temp_val1 = temp_val % (3*cfg.xbar_size)
                if (temp_val1 < cfg.xbar_size):
                    xbar_type = 'f'
                elif (temp_val1 < 2*cfg.xbar_size):
                    xbar_type = 'b'
                elif (temp_val1 < 3*cfg.xbar_size):
                    xbar_type = 'd'
                else:
                    assert (1==0), "xbar memory addressing failed"

            if (cfg.inference):
                # find i or o
                if (data_addr < cfg.num_matrix*1*cfg.xbar_size):
                    mem_addr = 0
                else:
                    mem_addr = cfg.xbar_size

                # find xbar_addr
                xbar_addr = data_addr % cfg.xbar_size

                # find matrix_addr
                num_matrix = (data_addr // (1*cfg.xbar_size)) % cfg.num_matrix

                # find xbar_type
                temp_val = (data_addr % (cfg.num_matrix*1*cfg.xbar_size))
                temp_val1 = temp_val % (1*cfg.xbar_size)
                if (temp_val1 < cfg.xbar_size):
                    xbar_type = 'f'
                else:
                    assert (1==0), "xbar memory addressing failed"   
                    
            return [num_matrix, xbar_type, mem_addr, xbar_addr]

        # write to the xbar memory (in/out) space depending on the address
        def writeToXbarMem (self, data_addr, data):
            [matrix_id, xbar_type, mem_addr, xbar_addr] = getXbarAddr (data_addr)
            if (mem_addr < cfg.xbar_size):
                # this is the xbarInMem
                self.xb_inMem_list[matrix_id][xbar_type].write (xbar_addr, data)
            else:
                # this is the xbarOutMem
                self.xb_outMem_list[matrix_id][xbar_type].write_n (xbar_addr,data)

        # read from xbar memory (in/out) depending on the address
        def readFromXbarMem (self, data_addr):
            [matrix_id, xbar_type, mem_addr, xbar_addr] = getXbarAddr (data_addr)
            if (mem_addr < cfg.xbar_size):
                # this is the xbarInMem
                return self.xb_inMem_list[matrix_id][xbar_type].read_n (xbar_addr)
            else:
                # this is the xbarOutMem
                return self.xb_outMem_list[matrix_id][xbar_type].read (xbar_addr)

        # Define what to do in execute (done for conciseness)
        def do_execute (self, ex_op, fid):

            if (ex_op == 'ld'):
                self.ldAccess_done = 0
                data = self.mem_interface.ramload
                # based on the address write to dataMem or xb_inMem
                data_addr = self.de_d1 + self.ex_vec_count * self.de_r2
                # check if data is a list
                if (type(data) != list):
                    data = ['0'*cfg.data_width]*self.de_r2
                for i in range (self.de_r2):
                    dst_addr = data_addr + i
                    if (dst_addr >= datamem_off):
                        try:
                            self.dataMem.write (dst_addr, data[i])
                        except:
                            print(data)
                            print((len(data)))
                            print(i)
                            print((self.ima_id))
                            print((self.de_d1))
                            print((self.ex_vec_count))
                            print((self.de_r1))
                            print((self.de_r2))
                            print((self.de_vec))
                            print((self.mem_interface.wait))
                            assert(0)
                            
                    else:
                        writeToXbarMem (self, dst_addr, data[i])

            elif (ex_op == 'st'): #nothing to be done by ima for st here
                return 1

            elif (ex_op == 'set'):
                # Updated for separate data_width and addr_width
                assert(self.de_d1 >= datamem_off), "set instruction cannot write to MVMU buffer"
                set_type = 'addr' if self.de_r1 else 'data'
                value=self.de_val1
                if set_type == 'data':
                    value = float2fixed(value, datacfg.int_bits, datacfg.frac_bits)
                else:
                    value = int2bin(value, cfg.addr_width)
                for i in range (self.de_vec):
                    # write to dataMem - check if addr is a valid datamem address
                    dst_addr = self.de_d1 + i
                    assert(dst_addr < cfg.dataMem_size + datamem_off), "Exceeded Data Memory Size"
                    self.dataMem.write(addr=dst_addr, data=value, type_t=set_type)

            elif (ex_op == 'cp'):
                for i in range (self.de_vec):
                    src_addr = self.de_r1 + i
                    # based on address read from dataMem or xb_inMem
                    if (src_addr >= datamem_off):
                        ex_val1 = self.dataMem.read (src_addr)
                    else:
                        ex_val1 = readFromXbarMem (self, src_addr)

                    dst_addr = self.de_d1 + i
                    # based on the address write to dataMem or xb_inMem
                    if (dst_addr >= datamem_off):
                        self.dataMem.write (dst_addr, ex_val1)
                    else:
                        writeToXbarMem (self, dst_addr, ex_val1)

            elif (ex_op == 'alu'):
                for i in range (self.de_vec):
                    # read val 1 either from data memory or xbar_outmem
                    src_addr1 = self.de_r1 + i
                    if (src_addr1 >= datamem_off):
                        ex_val1 = self.dataMem.read (src_addr1)
                    else:
                        ex_val1 = readFromXbarMem (self, src_addr1)

                    # read val 2 either from data memory or xbar_outmem
                    src_addr2 = self.de_r2 + i
                    if (src_addr2 >= datamem_off):
                        ex_val2 = self.dataMem.read (src_addr2)
                    else:
                        ex_val2 = readFromXbarMem (self, src_addr2)

                    # compute in ALU
                    [out, ovf] = self.alu_list[0].propagate (ex_val1, ex_val2, self.de_aluop, self.de_val1) #self.de_val1 is the 3rd operand for lsh
                    if (ovf):
                        fid.write ('IMA: ' + str(self.ima_id) + ' ALU Overflow Exception ' +\
                                self.de_aluop + ' allowed to run')

                    # write to dataMem - check if addr is a valid datamem address
                    dst_addr = self.de_d1 + i

                    if (dst_addr >= datamem_off):
                        self.dataMem.write (dst_addr, out)
                    else:
                        assert (0), "ALU instructions cannot write to xbar memory"
                        writeToXbarMem (self, dst_addr, ex_val1)

            elif (ex_op == 'alui'):
                for i in range (self.de_vec):
                    # read val 2 either from data memory or xbar_outmem
                    src_addr2 = self.de_r1 + i
                    if (src_addr2 >= datamem_off):
                        ex_val2 = self.dataMem.read (src_addr2)
                    else:
                        ex_val2 = readFromXbarMem (self, src_addr2)

                    # compute in ALU
                    [out, ovf] = self.alu_list[0].propagate (self.de_val1, ex_val2, self.de_aluop)
                    if (ovf):
                        fid.write ('IMA: ' + str(self.ima_id) + ' ALU Overflow Exception ' +\
                                self.de_aluop + ' allowed to run')

                    # write to dataMem - check if addr is a valid datamem address
                    dst_addr = self.de_d1 + i
                    if (dst_addr >= datamem_off):
                        self.dataMem.write (dst_addr, out)
                    else:
                        assert (0), "ALU instructions cannot write to xbar memory"
                        writeToXbarMem (self, dst_addr, ex_val1)

            elif (ex_op == 'mvm'):
                ## Define function to perform inner-product on specified mvmu
                # Note: Inner product with shift and add (shift-sub with last bit), works for 2s complement
                # representation for positive and negative numbers
                @profile
                def inner_product (mat_id, key):
                    # test if this is the tracking xbar
                    tracking_this = False
                    if tracking and (mat_id == track_mat_id) and (key == track_type):
                        print(("Tracking matrix %s, %d, addr = %d" % (key, mat_id, track_addr)))
                        print('')
                        tracking_this = True
                    
                    # reset the xb out memory before starting to accumulate
                    self.xb_outMem_list[mat_id][key].reset ()

                    sparsity=0
                    sparsity_adc=0
                    if cfg.sparse_opt:
                        xbar_inMem = self.xb_inMem_list[mat_id][key].read_all ()
                        non_0_val = 0
                        for i in range(cfg.xbar_size):
                            if xbar_inMem[i] != '0000000000000000':
                                non_0_val = non_0_val +1
                        sparsity = int((cfg.xbar_size-non_0_val)*100.0/cfg.xbar_size)
                        sparsity_adc = sparsity
                        if (sparsity%10!=0):
                            sparsity = sparsity-(sparsity%10)
                        else:
                            if (sparsity == 100):
                                sparsity = sparsity-10
                    
                    '''
                    # Quick calculation optimization
                    # we calculate all the mvm at the same time and let numpy fully use the parallelism
                    input_list = []
                    matrix_list_pos = []
                    matrix_list_neg = []
                    for k in range (int(math.ceil(cfg.input_prec / cfg.dac_res))):
                        out_xb_inMem = self.xb_inMem_list[mat_id][key].read (cfg.dac_res)
                        for m in range (datacfg.ReRAM_xbar_num):
                            input_list.append(self.dacArray_list[mat_id][key].propagate(out_xb_inMem)) #pass through
                            [matrix_pos, matrix_neg] = self.matrix_list[mat_id][key][m].get_value()
                            matrix_list_pos.append(matrix_pos)
                            matrix_list_neg.append(matrix_neg)
                    
                    input_list = np.array(input_list)
                    matrix_list_pos = np.array(matrix_list_pos)
                    matrix_list_neg = np.array(matrix_list_neg)

                    result_pos = np.einsum('ij,ijk->ik', input_list, matrix_list_pos)
                    result_neg = np.einsum('ij,ijk->ik', input_list, matrix_list_neg)
                    '''

                    ## Loop to cover all bits of inputs
                    for k in range (int(math.ceil(cfg.input_prec / cfg.dac_res))): #quantization affects the # of streams
                    #for k in range (1):
                        # read the values from the xbar's input register
                        out_xb_inMem = self.xb_inMem_list[mat_id][key].read (cfg.dac_res)
                        
                        #*************************************** HACK *********************************************
                        ###### CAUTION: Not replicated exact "functional" circuit behaviour for analog parts
                        ###### Use propagate (not propagate_hack) for DAC, Xbar, TIA, SNH, ADC when above is done
                        #*************************************** HACK *********************************************

                        # convert digital values to analog
                        out_dac = self.dacArray_list[mat_id][key].propagate(out_xb_inMem) #pass through

                        # Do MVM in each xbar (weight is distributed)
                        out_xbar_pos = [[] for x in range(datacfg.ReRAM_xbar_num)]
                        out_xbar_neg = [[] for x in range(datacfg.ReRAM_xbar_num)]
                        out_snh_pos = [[] for x in range(datacfg.ReRAM_xbar_num)]
                        out_snh_neg = [[] for x in range(datacfg.ReRAM_xbar_num)]

                        for m in range (datacfg.ReRAM_xbar_num):
                            # compute dot-product
                            [out_xbar_pos[m], out_xbar_neg[m]] = self.matrix_list[mat_id][key][m].propagate(out_dac, sparsity)
                            # do sampling
                            self.snh_list_pos[mat_id * datacfg.ReRAM_xbar_num + m].propagate(out_xbar_pos[m])
                            self.snh_list_neg[mat_id * datacfg.ReRAM_xbar_num + m].propagate(out_xbar_neg[m])
                            #self.snh_list_pos[mat_id * datacfg.ReRAM_xbar_num + m].propagate(result_pos[k * datacfg.ReRAM_xbar_num + m])
                            #self.snh_list_neg[mat_id * datacfg.ReRAM_xbar_num + m].propagate(result_neg[k * datacfg.ReRAM_xbar_num + m])
                            # reads out from sample&hold 
                            # NOTE: theoretically this should be done in the next loop. to minimize the change and simplify the code I put it here
                            out_snh_pos[m] = self.snh_list_pos[mat_id * datacfg.ReRAM_xbar_num + m].read()
                            out_snh_neg[m] = self.snh_list_neg[mat_id * datacfg.ReRAM_xbar_num + m].read()

                        # each of the xbar produce shifted bits of output (weight bits have been distributed)
                        for j in range (cfg.xbar_size): # this 'for' across xbar outs to adc happens via mux
                            out_sna = 0 # a zero for first sna
                            for m in range (datacfg.ReRAM_xbar_num):
                                # convert from analog to digital
                                adc_id = j // cfg.num_column_per_adc

                                # The commanted code below is the original code, shows how hardwarw should work
                                # Here to make the code run faster, we directly use the output of SNH.
                                # This is not the exact hardware behaviour, but it should work exactly the same
                                # Also use propagate_dummy to count the times mux is called
                                #out_mux_pos = self.mux_list[mat_id][key][m][adc_id]['pos'].propagate(out_snh_pos[m][adc_id * cfg.num_column_per_adc: (adc_id + 1) * cfg.num_column_per_adc], j % cfg.num_column_per_adc)
                                #out_mux_neg = self.mux_list[mat_id][key][m][adc_id]['neg'].propagate(out_snh_neg[m][adc_id * cfg.num_column_per_adc: (adc_id + 1) * cfg.num_column_per_adc], j % cfg.num_column_per_adc)
                                out_mux_pos = out_snh_pos[m][j]
                                out_mux_neg = out_snh_neg[m][j]
                                self.mux_list[mat_id][key][m][adc_id]['pos'].propagate_dummy()
                                self.mux_list[mat_id][key][m][adc_id]['neg'].propagate_dummy()

                                out_adc = '0' * cfg.adc_res
                                if self.adc_type == 'normal':
                                    out_adc_pos = self.adc_list[mat_id][key][m][adc_id]['pos'].propagate(out_mux_pos, datacfg.bits_per_cell[m], cfg.dac_res, sparsity_adc, return_type = 'int')
                                    out_adc_neg = self.adc_list[mat_id][key][m][adc_id]['neg'].propagate(out_mux_neg, datacfg.bits_per_cell[m], cfg.dac_res, sparsity_adc, return_type = 'int')
                                    # NOTE here to deal with overflow, we use propagate_float. this should be fixed later
                                    [out_adc, ovf] = self.alu_list[0].propagate(out_adc_pos, out_adc_neg, 'sub', return_type = 'float')
                                elif self.adc_type == 'differential':
                                    out_adc = self.adc_list[mat_id][key][m][adc_id].propagate(out_mux_pos, out_mux_neg, datacfg.bits_per_cell[m], cfg.dac_res, sparsity_adc)
                                    out_adc = bin2int(out_adc, cfg.adc_res)

                                if tracking_this and (j == track_addr):
                                    print(("xbar ID: %d, digit: %d" % (m, k + 1)))
                                    print(("out_mux_pos: %f, out_mux_neg: %f" % (out_mux_pos, out_mux_neg)))
                                    if self.adc_type == 'normal':
                                        print(("out_adc_pos: %d, out_adc_neg: %d" % (bin2int(out_adc_pos, 8), bin2int(out_adc_neg, 8))))
                                    print(("out_adc: %f" % out_adc))
                                    print(("value before sna: %f" % out_sna))

                                # Do the shift and add for mth xbar
                                [out_sna, ovf] = self.alu_list[0].propagate(out_sna, out_adc, 'sna', datacfg.stored_bit[m] - datacfg.frac_bits, return_type = 'float')

                                if tracking_this and (j == track_addr):
                                    print(("value after sna: %f" % out_sna))
                                    print('')

                            # read from xbar's output register
                            out_xb_outMem = self.xb_outMem_list[mat_id][key].read (j)
                            # shift and add - make a dedicated sna unit -- PENDING
                            [out_sna, ovf] = self.alu_list[0].propagate(out_xb_outMem, out_sna, 'sna', k * cfg.dac_res - datacfg.frac_bits)

                            if tracking_this and (j == track_addr):
                                print(("before this digit: %f" % fixed2float(out_xb_outMem, datacfg.int_bits, datacfg.frac_bits)))
                                print(("after this digit: %f" % fixed2float(out_sna, datacfg.int_bits, datacfg.frac_bits)))
                                print('')
                            if (cfg.debug and ovf):
                                fid.write ('IMA: ' + str(self.ima_id) + ' ALU Overflow Exception ' +\
                                        self.de_aluop + ' allowed to run')
                            # store back to xbar's output register & restart it
                            self.xb_outMem_list[mat_id][key].write (out_sna)
                        self.xb_outMem_list[mat_id][key].restart()

                    # stride the inputs if applicable
                    self.xb_inMem_list[mat_id][key].stride(self.de_val1, self.de_val2)



                ## Define function to perform outer-product on specified mvmu
                # NOTE: outer_product uses signed magnitude representations for positive and negative numbers
                # TODO: for training this function is needed, not modified yet.
                def outer_product (mat_id, key):
                    # read the bw-error to provide inputs across columns - read_a needs an energy/latency model - needs UPDATE
                    out_xb_outMem = self.xb_outMem_list[mat_id][key].read_p() # read entire xb_outMem

                    # Loop to cover all bits of inputs - bit-streamed inputs across rows
                    for j in range (cfg.xbdata_width/cfg.dac_res):

                        # read the fw-activations to provide inputs across the rows
                        out_xb_inMem = self.xb_inMem_list[mat_id][key].read (cfg.dac_res)

                        # left shift the bw-error values for subsequent bit-streamed computation (jth loop) to make a list of
                        # 32-bit values
                        out_xb_outMem_temp = [((cfg.num_bits-j)*'0' + val + j*'0') for val in out_xb_outMem]

                        # do outer product on all physical xbars (for a logical xbar)
                        # Note: 2X delta xbars than fw/bw xbars
                        num_xb = (2*cfg.data_width) / cfg.xbar_bits
                        for m in range (num_xb):
                            out_dac1 = self.dacArray_list[mat_id]['d_r'].propagate (out_xb_inMem)
                            if (m == 0):
                                temp = [val[-((m+1)*cfg.xbar_bits):] for val in out_xb_outMem_temp]
                            else:
                                temp = [val[-((m+1)*cfg.xbar_bits):-(m*cfg.xbar_bits)] for val in out_xb_outMem_temp]
                            out_dac2 = self.dacArray_list[mat_id]['d_c'].propagate (temp)

                            self.matrix_list[mat_id][key][m].propagate_op (out_dac1, out_dac2, cfg.lr, cfg.dac_res, self.matrix_list[mat_id][key][m].bits_per_cell)

                ## Traverse through the matrices in a core
                if (cfg.training):
                    for i in range (cfg.num_matrix):
                    # traverse through f/b/d mvmu(s) for the matrix and execute if applicable
                        mask_temp = self.de_xb_nma[i]
                        if (mask_temp[0] == '1'):
                        # foward xbar operation
                            #print ("ima_id: " + str(self.ima_id) + " mat_id: "  + str(i) + " MVM")
                            inner_product (i, 'f')
                        if (mask_temp[1] == '1'):
                        #print ("ima_id: " + str(self.ima_id) + " mat_id: "  + str(i) + " MTVM")
                        # backward xbar operation
                            inner_product (i, 'b')
                        if (mask_temp[2] == '1'):
                            outer_product (i, 'd')

                if (cfg.inference):
                   for i in range(cfg.num_matrix):
                       if self.de_xb_nma[i]:
                           #print ("ima_id: " +str(self.ima_id) + " mat_id: "  +str(i) + " MVM")
                           inner_product(i,'f')

            # TODO for training this need to modify. not supporting negative param yet
            elif (ex_op == 'crs'):
                # read weights from delta-xbar, synchronize, write to f/b xbars
                num_xbD = 2 * datacfg.ReRAM_xbar_num
                num_xbF = datacfg.ReRAM_xbar_num
                for mat_id in range (cfg.num_matrix):
                    mask_temp = self.de_xb_nma[mat_id]
                    if (mask_temp == '1'):
                        for k in range (cfg.xbar_size):
                            for l in range (cfg.xbar_size):
                                # read wt slices from delta xbar to compose a new weight
                                wt_new_float = 0.0
                                for m in range (num_xbD):
                                    wt_new_float += self.matrix_list[mat_id]['d'][2 * m].read(k,l) * (2 ** (2 * datacfg.stored_bit[m] - datacfg.frac_bits)) # left shift
                                    wt_new_float += self.matrix_list[mat_id]['d'][2 * m + 1].read(k,l) * (2 ** (2 * datacfg.stored_bit[m] + datacfg.storage_config[m] - datacfg.frac_bits)) # left shift
                                # write wt slices to f and b xbar
                                # captures precision loss, as values read from 16 xbars (32-bits) are converted to 16-bits
                                wt_new_fixed = float2fixed (wt_new_float, datacfg.int_bits, datacfg.frac_bits)
                                for m in range (num_xbF):               
                                    if (m == 0):
                                        val = wt_new_fixed[-1 * datacfg.stored_bit(m + 1):]
                                    elif m == (num_xbF - 1):
                                        val = wt_new_fixed[:datacfg.bits_per_cell[m]]
                                        # augment sign extension (used in MSB xbar only)
                                        val = (datacfg.num_bits - datacfg.stored_bit[m])*val[0] + val[0:]
                                    else:
                                        val = wt_new_fixed[-1 * datacfg.stored_bit(m + 1): -1 * datacfg.stored_bit(m + 1) + datacfg.bits_per_cell[m]]
                                    
                                    val_float = fixed2float(val, datacfg.int_bits, datacfg.frac_bits) # xbar_value in xbar stores float values
                                    self.matrix_list[mat_id]['f'][m].write(k, l, val_float)
                                    self.matrix_list[mat_id]['b'][m].write(k, l, val_float)

            elif (ex_op == 'jmp'):
                self.fd_instrn['opcode'] = 'nop'
                self.pc = self.de_instrn['imm']

            elif (ex_op == 'beq'):
                [out, ovf] = self.alu_int.propagate (self.de_val1, self.de_val2, self.de_aluop) #self.de_val1 is the 3rd operand for lsh
                out_int = bin2int(out, cfg.num_bits)
                if (out_int == 1):
                    # should add a mux unit (for realistic hw here to update pc & pipe registers)
                    self.fd_instrn['opcode'] = 'nop'
                    self.pc = self.de_instrn['imm']

            elif (ex_op == 'alu_int'): # produces values used by load/st (mem addr read from dataMem), beq (operand reads)
                [out, ovf] = self.alu_int.propagate (self.de_val1, self.de_val2, self.de_aluop) #self.de_val1 is the 3rd operand for lsh
                # write to dataMem - check if addr is a valid datamem address
                assert (self.de_d1 >= datamem_off), 'ALU instrn: datamemory write addrress is invalid'
                self.dataMem.write (self.de_d1, out)

            elif (ex_op == 'hlt'): # for halt instruction
                self.halt = 1
            # do nothing for nop instruction

        # Computes the latency for Analog mvm instruction based on DPE configuration
        def xbComputeLatency_Analog (self, mask):
            latency_out_list = []
            fb_found = 0
            d_found = 0
            latency_out_list = []
            for idx, temp in enumerate(mask):
                #print("idx", idx)
                if ((temp[0] == '1') or (temp[1] == '1')):
                    fb_found += 1
                    #break
                if (temp[2] == '1'):
                    d_found += 1
                    #break

                ## MVM inner product goes through a 3 stage pipeline (each stage consumes 128 cycles - xbar aces latency)
                # Cycle1 - xbar_inMem + DAC + XBar
                # Cycle2 - SnH + ADC
                # Cycle3 - SnA + xbar_outMem
                # The above pipeline is valid for one ADC per physical xbar only !! (Update for other cases, if required)
                num_stage = 3
                #lat_temp = self.matrix_list[0]['f'][0].getIpLatency() # due to xbar access
                lat_temp = 0
                # We assume all ADCs in a matrix has the same resolution
                #adc_idx = idx*cfg.num_adc_per_matrix
                if cfg.adc_type == 'normal':
                    lat_temp = self.adc_list[0]['f'][0][0]['pos'].getLatency()
                elif cfg.adc_type == 'differential':
                    lat_temp = self.adc_list[0]['f'][0][0].getLatency()
                '''
                print("adc_idx", adc_idx)
                print("lat_temp", lat_temp)
                print("self.adc_list[adc_idx].adc_res", self.adc_list[adc_idx].adc_res)
                for adccccc in self.adc_list:
                    print("adccccc.adc_res", adccccc.adc_res)
                print("---")
                '''
                latency_ip = lat_temp * ((cfg.input_prec / cfg.dac_res) + num_stage - 1) * float(int(fb_found>0))
                #latency_ip = lat_temp * ((cfg.input_prec / cfg.dac_res) + num_stage - 1) * float(int(fb_found>0))*(math.ceil(float(cfg.weight_width)/ \
                #cfg.xbar_bits) /math.ceil(float(cfg.data_width)/cfg.xbar_bits)) # last term to account for the effect of quantization on latency
                ## MVM outer product occurs in 4 cycles to take care of all i/o polarities (++, +-, -+, --)
                num_phase = 4
                lat_temp = 0
                for m in range(datacfg.ReRAM_xbar_num):
                    lat_temp = max(self.matrix_list[0]['f'][m].getOpLatency(), lat_temp)
                #latency_op = lat_temp * num_phase * d_found
                latency_op = lat_temp * num_phase * float(int(d_found>0))
                ## output latency should be the max of ip/op operation
                latency_out = max(latency_ip, latency_op)
                #print ("Mask", mask)
                #print ("Latency IP", latency_ip)
                #print ("Latency OP", latency_op)
                #print ("latency_out", latency_out)
                latency_out_list.append(latency_out)
            return max(latency_out_list)

        # Computes the latency for Analog mvm instruction based on DPE configuration
        def xbComputeLatency_Digital (self):
            mvm_lat_temp = 0
            if (cfg.inference):
                for p in range(cfg.num_matrix):
                    if self.de_xb_nma[p]:
                        sparsity=0
                        if cfg.sparse_opt:
                            xbar_inMem = self.xb_inMem_list[p]['f'].read_all ()
                            non_0_val = 0
                            for i in range(cfg.xbar_size):
                                if xbar_inMem[i] != '0000000000000000':
                                    non_0_val = non_0_val +1
                            sparsity = int((cfg.xbar_size-non_0_val)*100.0/cfg.xbar_size)
                            if (sparsity%10!=0):
                                sparsity = sparsity-(sparsity%10)
                            else:
                                if (sparsity == 100):
                                    sparsity = sparsity-10
                        mvm_lat_temp += digi_param.Digital_xbar_lat_dict[cfg.MVMU_ver][str(cfg.xbar_size)][str(sparsity)]
            return mvm_lat_temp

        # State machine runs only if the stage is non-empty
        # Describe the functionality on a cycle basis
        if (self.stage_empty[sId] != 1):
            # First cycle - update the target latency
            if (self.stage_cycle[sId] == 0):
                # Check for assertion pass
                ex_op = self.de_opcode
                assert (ex_op in param.op_list), 'unsupported opcode'

                # assign execution unit based stage latency
                if (ex_op in ['ld', 'st']):
                    if (ex_op == 'ld'):
                        self.stage_latency[sId] = self.mem_interface.getLatency() #mem_interface has infinite latency
                        self.mem_interface.rdRequest (self.de_r1 + self.ex_vec_count * self.de_r2, self.de_r2)
                    elif (ex_op == 'st'):
                        self.stage_latency[sId] = self.dataMem.getLatency() #mem_interface has infinite latency

                elif (ex_op == 'cp'):
                    # cp instructions reads from datamemory/xbinmem & writes to xb_inmem/datamem
                    unit_lat = self.dataMem.getLatency()
                    #self.stage_latency[sId] = self.de_vec * unit_lat
                    self.stage_latency[sId] = unit_lat # cp can just assign mux selectors for each xbar (which inmem feeds the xbar)

                elif (ex_op == 'set'):
                    # set writes to data memory
                    unit_lat = self.dataMem.getLatency()
                    self.stage_latency[sId] = self.de_vec * unit_lat

                elif (ex_op == 'alu' or ex_op == 'alui'):
                    # ALU instructions read from memory, access ALU and write to memory
                    unit_lat = self.alu_list[0].getLatency ()
                    #unit_lat = self.dataMem.getLatency() + \
                    #            self.alu_list[0].getLatency() + self.dataMem.getLatency()
                    self.stage_latency[sId] = int (math.ceil(self.de_vec / cfg.num_ALU)) * unit_lat

                elif (ex_op == 'mvm'):
                    mask_temp = self.de_xb_nma
                    if (cfg.MVMU_ver == "Analog"):
                        self.stage_latency[sId] = xbComputeLatency_Analog (self, mask_temp) # mask tells which of ip/op or both is occurring
                    else:
                        self.stage_latency[sId] = xbComputeLatency_Digital(self)

                # Needs update - use xbar serial read latency
                elif (ex_op == 'crs'):
                    temp_lat = 0
                    for m in datacfg.ReRAM_xbar_num:
                        temp_lat = max(self.matrix_list[0]['f'][m].getWrLatency(), temp_lat)
                        temp_lat = max(self.matrix_list[0]['f'][m].getRdLatency(), temp_lat)
                    
                    self.stage_latency[sId] = temp_lat

                elif (ex_op in ['beq', 'alu_int']):
                    self.stage_latency[sId] = self.alu_int.getLatency ()

                else: # halt/jmp/nop instruction
                    self.stage_latency[sId] = 1

                # Check if first = last cycle - NA for LD/ST
                # (EDRAM + Controller always latency >= 2) - Follow this else deisgn breaks
                if (ex_op == 'st' and self.stage_latency[sId] == 0):
                    # read the data from dataMem or xb_outMem depending on address
                    st_data_addr =  self.de_r1 + self.ex_vec_count * (self.de_r2) # address of data in register HERE!!!
                    ex_val1 = ['' for num in range (self.de_r2)] # modified
                    if (st_data_addr >= cfg.num_xbar * cfg.xbar_size):
                        for num in range (self.de_r2): # modified
                            ex_val1[num] = self.dataMem.read (st_data_addr+num) # modified
                    else:
                        xb_id = st_data_addr / cfg.xbar_size
                        addr = st_data_addr % cfg.xbar_size
                        for num in range (self.de_r2): # modified
                            ex_val1[num] = self.xb_outMem_list[xb_id].read (addr+num) # modified
                    # combine counter and data
                    ramstore = [str(self.de_val1), ex_val1[:]] # modified - 1st item in list: counter value, 2nd item: list of values to be written to edram
                    self.mem_interface.wrRequest (self.de_d1 + \
                            self.ex_vec_count * self.de_r2, ramstore, self.de_r2)
                    # to make sure st looks for memwait after datamem read
                    self.stage_cycle[sId] = self.stage_cycle[sId] + 1

                elif (ex_op != 'st' and self.stage_latency[sId] == 1 and update_ready): # NA for LD/ST
                    do_execute (self, ex_op, fid)
                    self.stage_done[sId] = 1
                    self.stage_cycle[sId] = 0
                    self.stage_empty[sId] = 1

                else: # NA for LD/ST
                    self.stage_cycle[sId] = self.stage_cycle[sId] + 1

            # Check whether datamem access for st has finished
            elif (self.de_opcode == 'st' and self.stage_cycle[sId] == self.stage_latency[sId]):
                # read the data from dataMem or xb_outMem depending on address
                st_data_addr =  self.de_r1 + self.ex_vec_count * (self.de_r2) # address of data in register
                ex_val1 = ['' for num in range (self.de_r2)] # modified
                if (st_data_addr >= datamem_off):
                    for num in range (self.de_r2): # modified
                        ex_val1[num] = self.dataMem.read (st_data_addr+num) # modified
                        if len(ex_val1[num]) != cfg.data_width:
                            print("Error: data width mismatch")
                            print(("data width: ", len(ex_val1[num])))
                            print(("st_data_addr: ", st_data_addr))
                            print(("num: ", num))
                            print(("ex_val1[num]: ", ex_val1[num]))
                            print(("IMA ID: ", self.ima_id))
                            print(("width:", self.de_r2))
                            assert(0)
                else:
                    for num in range (self.de_r2): # modified
                        ex_val1[num] = readFromXbarMem (self, st_data_addr+num)
                # combine counter and data
                ramstore = [str(self.de_val1), ex_val1[:]] # modified - 1st item in list: counter value, 2nd item: list of values to be written to edram
                self.mem_interface.wrRequest (self.de_d1 + self.ex_vec_count * self.de_r2, ramstore, self.de_r2)
                # to make sure st looks for memwait after datamem read
                self.stage_cycle[sId] = self.stage_cycle[sId] + 1

            # Last cycle - update pipeline registers (if ??) & done flag - or condition is for LD/ST
            elif (((not self.de_opcode in ['ld', 'st']) and self.stage_cycle[sId] >= self.stage_latency[sId]-1 and update_ready) or \
                  (self.de_opcode == 'st' and self.mem_interface.wait == 0 and self.ex_vec_count == (self.de_vec-1) and update_ready) or \
                  (self.de_opcode == 'ld' and self.stage_cycle[sId] >= self.stage_latency[sId]-1 and self.ex_vec_count == (self.de_vec-1) and update_ready)):
                ex_op = self.de_opcode
                #print ("doing exe stage for op: " + ex_op)
                do_execute (self, ex_op, fid)
                self.stage_done[sId] = 1
                self.stage_cycle[sId] = 0
                self.stage_empty[sId] = 1
                self.ex_vec_count = 0

            # For LD and ST when all units until last vector
            elif ((self.de_opcode == 'ld' and self.stage_cycle[sId] >= self.stage_latency[sId]-1) or \
                    (self.de_opcode == 'st' and self.mem_interface.wait == 0)):
                ex_op = self.de_opcode
                do_execute (self, ex_op, fid)
                self.stage_cycle[sId] = 0
                self.ex_vec_count += 1

            # For all other cycles
            else:
                # Assumption - DataMemory cannot be done in the last edram access cycle
                if (self.de_opcode == 'ld' and self.mem_interface.wait == 0 and self.ldAccess_done == 0): # LD finishes after mem_access + reg_write is done
                    self.ldAccess_done = 1
                    self.stage_cycle[sId] = self.stage_latency[sId] - self.dataMem.getLatency () # can be data_mem too
                else:
                    self.stage_cycle[sId] = self.stage_cycle[sId] + 1


    #####################################################
    ## Define how pipeline executes
    #####################################################
    def pipe_init (self, instrn_filepath, fid = ''):
        self.debug = 0
        # tracefile stores the debug trace in debug mode
        if (cfg.debug and (fid != '')):
            self.debug = 1
            fid.write ('Cycle information is printed is at the end of the clock cycle\n')
            fid.write ('Assumption: A clock cycle ends at the positive edge\n')

        self.halt = 0

        zero_list = [0] * self.num_stage
        one_list = [1] * self.num_stage

        self.stage_empty = one_list[:]
        self.stage_empty[0] = 0 # fetch doesn't begin with empty
        self.stage_cycle = zero_list[:]
        self.stage_done = one_list[:]

        #Initialize the instruction memory
        dict_list = np.load(instrn_filepath, allow_pickle=True)
        self.instrnMem.load(dict_list)

        self.ldAccess_done = 0
        self.cycle_count = 0

    # Mimics one cycle of ima pipeline execution
    def pipe_run (self, cycle, fid = ''): # fid is tracefile's id
        self.cycle_count += 1
        # Run the pipeline for once cycle
        # Define a stage function
        stage_function = {0 : self.fetch,
                          1 : self.decode,
                          2 : self.execute}

        # Traverse the pipeline to update the update_ready flag & execute the stages in backward order
        for i in range (self.num_stage-1, -1, -1):
            # set update_ready flag
            if (i == self.num_stage-1):
                update_ready = 1
            else:
                update_ready = self.stage_done[i+1]

            # run the stage based on its update_ready argument
           
            stage_function[i] (update_ready, fid)

        # If specified, print thetrace (pipeline stage information)
        if (self.debug):
            fid.write('Cycle ' + str(cycle) + '\n')

            sId = 0 # Fetch
            fid.write('Fet | PC ' + str(self.pc))
            fid.write(' | Flags: empty ' + str(self.stage_empty[sId]) + ' done ' + str(self.stage_done[sId]) \
                    + ' cycles ' + str(self.stage_cycle[sId]) + '\n')

            sId = 1 # Decode
            fid.write ('Dec | Inst: ')
            json.dump (self.fd_instrn, fid)
            fid.write(' | Flags: empty ' + str(self.stage_empty[sId]) + ' done ' + str(self.stage_done[sId]) \
                    + ' cycles ' + str(self.stage_cycle[sId]) + '\n')

            sId = 2 # Execute
            fid.write('Exe | Inst: ')
            json.dump(self.de_instrn, fid)
            fid.write ('curr_vec: ' + str (self.ex_vec_count))
            fid.write(' | Flags: empty ' + str(self.stage_empty[sId]) + ' done ' + str(self.stage_done[sId]) \
                    + ' cycles ' + str(self.stage_cycle[sId]) + '\n')
            fid.write('\n')

            if (self.halt == 1):
                fid.write ('IMA halted at ' + str(cycle) + ' cycles')
