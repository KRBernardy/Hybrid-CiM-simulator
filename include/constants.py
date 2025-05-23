## This file contains the data structures used in differnet hierarchies.
## It also holds power, area and latency numbers of different component used in DPE design
import config as cfg
import math
import constants_digital as digi_param
# Limits the number of cycles an IMA runs in case it doesn't halt
infinity = 10000000

#############################################################################################################
## Technology/Other constants for all the modules
#############################################################################################################
# IMA - folliwng parameters are not used currently, will be used when analog functionality is implemented
cycle_time = 1 # in nanoseconds (1ns)
vdd = 1.0
xbar_conductance_min = 0.0
xbar_conductance_max = 1.0 # think about this - ???

#############################################################################################################
## Define commonly used data structures
#############################################################################################################
# List of supported opcodes for tile
op_list_tile = ['send', 'receive', 'compute', 'halt']

# Instruction format for Tile
dummy_instrn_tile = {'opcode' : op_list_tile[0],
                     'mem_addr': 0,     # send/receive - edram_addr
                     'r1': 0,     # send-send_width, receive-receive_width
                     'r2': 0,     # send-target_addr, receive-counter
                     'vtile_id': 0, # send/receive-neuron_id
                     'ima_nma': '',      # compute - a bit for each ima
                     'vec': 0} # vector width

# List of supported opcodes/aluops for IMA - cp will copy data (from data memory of ima to xbarInmem)
op_list = ['ld', 'cp', 'st', 'set', 'nop', 'alu', 'alui', 'mvm', 'vvo', 'hlt', 'jmp', 'beq', 'alu_int', 'crs']
aluop_list = ['add', 'sub', 'sna', 'mul', 'sigmoid'] # sna is also used by mvm isntruction

# Instruction format for IMA
dummy_instrn = {'opcode' : op_list[0],      # instrn op
               'aluop'  : aluop_list[0],   # alu function
               'd1'     : 0,               # destination
               'r1'     : 0,               # operand1 (stride for mvm)
               'r2'     : 0,               # operand2
               'r3'     : 0,               # operand3 (shift)
               'vec'    : 0,               # vector width
               'imm'    : 0,               # immediate (scalar) data
               'xb_nma' : 0 }              # xbar negative-mask, a xbar evaluates if neg-mask = 1

# List of pipeline stages - in order for IMA
stage_list = ['fet', 'dec', 'ex']
last_stage = 'ex'

#############################################################################################################
# IMA Hierarchy parameters
    # Number of Xbars
    # Crossbar Size
    # Crossbar bits
    # Bit resolution of ADCs and DACs
    # Number of ADCs
    # Number of ALUs
    # Data memory size
    # Size of Xbar in/out memory (Register) is dependent on Xbar size and num_bits
    # Instruction memory size
#############################################################################################################

# IMA component latency/power/area dictionary (all values in ns, mw, mm2)

#ReRAM accuracy degradtion. Use ReRAM_read_sigma = 0 for no degradtion
ReRAM_read_sigma = 0

# XBAR - Models from ISAAC paper
xbar_lat_dict = {'2': {'32' : 32,   # first indexed by xbar_bits then by xbar_size
                       '64' : 64,
                       '128': 128,
                       '256': 256},
                 '4': {'32' : 32,
                       '64' : 64,
                       '128': 128,
                       '256': 256},
                 '6': {'32' : 32,
                       '64' : 64,
                       '128': 128,
                       '256': 256}}

xbar_pow_dict = {'2': {'32' : 0.01875,
                       '64' : 0.075,
                       '128': 0.3,
                       '256': 1.2},
                 '4': {'32' : 0.01875,
                       '64' : 0.075,
                       '128': 0.3,
                       '256': 1.2},
                 '6': {'32' : 0.01875,
                       '64' : 0.075,
                       '128': 0.3,
                       '256': 1.2}}

xbar_area_dict = {'2': {'32' : 1.5625 * 10**(-6),
                       '64' : 6.25 * 10**(-6),
                       '128': 2.5 * 10**(-5),
                       '256': 1.0 * 10**(-4)},
                  '4': {'32' : 1.5625 * 10**(-6),
                       '64' : 6.25 * 10**(-6),
                       '128': 2.5 * 10**(-5),
                       '256': 1.0 * 10**(-4)},
                  '6': {'32' : 1.5625 * 10**(-6),
                       '64' : 6.25 * 10**(-6),
                       '128': 2.5 * 10**(-5),
                       '256': 1.0 * 10**(-4)}}

## New values added for xbar MVM/MTVM, OP (parallel write), serial read/write
# the following is lumped power for xbar inner/outer-product - includes peripherals

#xbar_op_lat = 20.0*12.8 # with 4 VFUs
#xbar_op_pow = 4.44 * 3.27 / (12.8)

#xbar_ip_lat = 100.0
#xbar_ip_pow = (1.37*2.0) # xbar_ip_pow (includes all mvmu)
#xbar_ip_pow = (1.37*2.0) - 1.04 # xbar_ip_pow (includes all mvmu except ADC - uncomment num_access for ADC object)

# Note the read and write lat/pow are for entire xbar
#xbar_rd_lat = 328.0 * 1000 * (1/32.0)
#xbar_wr_lat = 351.0 * 1000 * (1/32.0)

# the following is lumped power for xbar rd/wr (for whole array) - includes peripherals
#xbar_rd_pow = 208.0 * 1000 * (1/32.0) / xbar_rd_lat
#xbar_wr_pow = 676.0 * 1000 * (1/32.0) / xbar_rd_lat


## the above are now dicts since we want them to be different for different cells

xbar_op_lat_dict = {'1': 20.0 * 12.8,
                    '2': 20.0 * 12.8,
                    '3': 20.0 * 12.8,
                    '4': 20.0 * 12.8,
                    '5': 20.0 * 12.8,
                    '6': 20.0 * 12.8,
                    '7': 20.0 * 12.8,
                    '8': 20.0 * 12.8}

xbar_op_pow_dict = {'1': 4.44 * 3.27 / (12.8),
                    '2': 4.44 * 3.27 / (12.8),
                    '3': 4.44 * 3.27 / (12.8),
                    '4': 4.44 * 3.27 / (12.8),
                    '5': 4.44 * 3.27 / (12.8),
                    '6': 4.44 * 3.27 / (12.8),
                    '7': 4.44 * 3.27 / (12.8),
                    '8': 4.44 * 3.27 / (12.8)}

xbar_ip_lat_dict = {'1': 100.0,
                    '2': 100.0,
                    '3': 100.0,
                    '4': 100.0,
                    '5': 100.0,
                    '6': 100.0,
                    '7': 100.0,
                    '8': 100.0}

xbar_ip_pow_dict = {'1': 1.37 * 2.0 - 1.04,
                    '2': 1.37 * 2.0 - 1.04,
                    '3': 1.37 * 2.0 - 1.04,
                    '4': 1.37 * 2.0 - 1.04,
                    '5': 1.37 * 2.0 - 1.04,
                    '6': 1.37 * 2.0 - 1.04,
                    '7': 1.37 * 2.0 - 1.04,
                    '8': 1.37 * 2.0 - 1.04}

xbar_rd_lat_dict = {'1': 328.0 * 1000 * (1 / 32.0) * 1,
                    '2': 328.0 * 1000 * (1 / 32.0) * 2,
                    '3': 328.0 * 1000 * (1 / 32.0) * 3,
                    '4': 328.0 * 1000 * (1 / 32.0) * 400,
                    '5': 328.0 * 1000 * (1 / 32.0),
                    '6': 328.0 * 1000 * (1 / 32.0),
                    '7': 328.0 * 1000 * (1 / 32.0),
                    '8': 328.0 * 1000 * (1 / 32.0)}

xbar_wr_lat_dict = {'1': 351.0 * 1000 * (1 / 32.0) * 1,
                    '2': 351.0 * 1000 * (1 / 32.0) * 2,
                    '3': 351.0 * 1000 * (1 / 32.0) * 3,
                    '4': 351.0 * 1000 * (1 / 32.0) * 400,
                    '5': 351.0 * 1000 * (1 / 32.0),
                    '6': 351.0 * 1000 * (1 / 32.0),
                    '7': 351.0 * 1000 * (1 / 32.0),
                    '8': 351.0 * 1000 * (1 / 32.0)}

xbar_rd_pow_dict = {'1': 208.0 * 1000 * (1 / 32.0) / xbar_rd_lat_dict['1'],
                    '2': 208.0 * 1000 * (1 / 32.0) / xbar_rd_lat_dict['2'],
                    '3': 208.0 * 1000 * (1 / 32.0) / xbar_rd_lat_dict['3'],
                    '4': 208.0 * 1000 * (1 / 32.0) / xbar_rd_lat_dict['4'],
                    '5': 208.0 * 1000 * (1 / 32.0) / xbar_rd_lat_dict['5'],
                    '6': 208.0 * 1000 * (1 / 32.0) / xbar_rd_lat_dict['6'],
                    '7': 208.0 * 1000 * (1 / 32.0) / xbar_rd_lat_dict['7'],
                    '8': 208.0 * 1000 * (1 / 32.0) / xbar_rd_lat_dict['8']}

xbar_wr_pow_dict = {'1': 676.0 * 1000 * (1 / 32.0) / xbar_wr_lat_dict['1'],
                    '2': 676.0 * 1000 * (1 / 32.0) / xbar_wr_lat_dict['2'],
                    '3': 676.0 * 1000 * (1 / 32.0) / xbar_wr_lat_dict['3'],
                    '4': 676.0 * 1000 * (1 / 32.0) / xbar_wr_lat_dict['4'],
                    '5': 676.0 * 1000 * (1 / 32.0) / xbar_wr_lat_dict['5'],
                    '6': 676.0 * 1000 * (1 / 32.0) / xbar_wr_lat_dict['6'],
                    '7': 676.0 * 1000 * (1 / 32.0) / xbar_wr_lat_dict['7'],
                    '8': 676.0 * 1000 * (1 / 32.0) / xbar_wr_lat_dict['8']}



# DAC - Discuss exact values with ISSAC authors
dac_lat_dict = {'1' : 1,
                '2' : 1,
                '4' : 1,
                '8' : 1,
                '16': 1}

dac_pow_dyn_dict = {'1' : 0.00350625,
                    '2' : 0.00350625,
                    '4' : 0.00350625,
                    '8' : 0.00350625,
                    '16': 0.00350625}

dac_pow_leak_dict = {'1' : 0.000390625,
                     '2' : 0.000390625,
                     '4' : 0.000390625,
                     '8' : 0.000390625,
                     '16': 0.000390625}

dac_area_dict = {'1' : 1.67 * 10**(-7),
                 '2' : 1.67 * 10**(-7),
                 '4' : 1.67 * 10**(-7),
                 '8' : 1.67 * 10**(-7),
                 '16': 1.67 * 10**(-7)}

# ADC - Discuss exact values with ISSAC authors
# ADC Values for including sparsity
adc_lat_dict = {'1' : 12.5,
                '2' : 25,
                '3' : 37.5,
                '4' : 50,
                '5' : 62.5,
                '6' : 75,
                '7' : 87.5,
                '8' : 100,
		'9' : 112.5,
                '10': 125,
                '11': 137.5,
                '12': 150,
                '13': 162.5,
                '14': 175,
                '15': 187.5,
                '16': 200}

adc_pow_dyn_dict = {'1' : 0.225,
                    '2' : 0.45,
                    '3' : 0.675,
                    '4' : 0.9,
                    '5' : 1.125,
                    '6' : 1.35,
                    '7' : 1.575,
                    '8' : 1.8,
		    '9' : 2.025,
                    '10': 2.25,
                    '11': 2.475,
                    '12': 2.7,
                    '13': 2.925,
                    '14': 3.15,
                    '15': 3.375,
                    '16': 3.6}

adc_pow_leak_dict = {'1' : 0.025,
                     '2' : 0.05,
                     '3' : 0.075,
                     '4' : 0.1,
                     '5' : 0.125,
                     '6' : 0.15,
                     '7' : 0.175,
                     '8' : 0.2,
		     '9' : 0.225,
                     '10': 0.25,
                     '11': 0.275,
                     '12': 0.3,
                     '13': 0.325,
                     '14': 0.35,
                     '15': 0.375,
                     '16': 0.4}

adc_area_dict = {'1' : 0.0012,
                 '2' : 0.0012,
                 '3' : 0.0012,
                 '4' : 0.0012,
                 '5' : 0.0012,
                 '6' : 0.0012,
                 '7' : 0.0012,
                 '8' : 0.0012,
		 '9' : 0.0012,
                 '10': 0.0012,
                 '11': 0.0012,
                 '12': 0.0012,
                 '13': 0.0012,
                 '14': 0.0012,
                 '15': 0.0012,
                 '16': 0.0012}

diff_adc_lat_dict = {'1' : 12.5,
                '2' : 25,
                '3' : 37.5,
                '4' : 50,
                '5' : 62.5,
                '6' : 75,
                '7' : 87.5,
                '8' : 100,
		'9' : 112.5,
                '16': 200}

diff_adc_pow_dyn_dict = {'1' : 0.225,
                    '2' : 0.45,
                    '3' : 0.675,
                    '4' : 0.9,
                    '5' : 1.125,
                    '6' : 1.35,
                    '7' : 1.575,
                    '8' : 1.8,
		    '9' : 2.025,
                    '16': 3.6}

diff_adc_pow_leak_dict = {'1' : 0.025,
                     '2' : 0.05,
                     '3' : 0.075,
                     '4' : 0.1,
                     '5' : 0.125,
                     '6' : 0.15,
                     '7' : 0.175,
                     '8' : 0.2,
		     '9' : 0.225,
                     '16': 0.4}

diff_adc_area_dict = {'1' : 0.0012,
                 '2' : 0.0012,
                 '3' : 0.0012,
                 '4' : 0.0012,
                 '5' : 0.0012,
                 '6' : 0.0012,
                 '7' : 0.0012,
                 '8' : 0.0012,
		 '9' : 0.0012,
                 '16': 0.0012}

# SNH (MVM pipeline)
snh_lat = 1
snh_pow_leak = 9.7 * 10**(-7)
snh_pow_dyn = 9.7 * 10**(-6) - snh_pow_leak
snh_area = 0.00004 / 8 / 128

# SNA (MVM pipeline)
sna_lat = 1
sna_pow_leak = 0.005
sna_pow_dyn = 0.05 - sna_pow_leak
sna_area = 0.00006

# ALU (Part of Vector Functional Unit)
alu_lat = 1
alu_pow_dyn = 2.4 * 32/45
alu_pow_div_dyn = 1.52 * 32/45
alu_pow_mul_dyn = 0.795 * 32/45
alu_pow_others_dyn = 0.373 * 32/45 # logical, eq, relu, add, sub, lsh, rsh
alu_pow_leak = 0.27 * 32/45
alu_area = 0.00567 * 32/45

# witout considering division
#alu_lat = 1
#alu_pow_dyn = 1.15 * 32/45
#alu_pow_mul_dyn = 0.796 * 32/45
#alu_pow_others_dyn = 0.36 * 32/45 # logical, eq, relu, add, sub, lsh, rsh
#alu_pow_leak = 0.05 * 32/45
#alu_area = 0.002326 * 32/45

# Sigmoid/Tanh (Part of Vector Functional Unit) - Taken from ISAAC paper
act_lat = 1 # added for 4 exponential units
act_pow_leak = 0.026
act_pow_dyn = 0.26 - act_pow_leak
act_area = 0.0003 # check this ???

# Multiplexer - These should be analog muxes
mux_lat = 0
mux_pow_leak = 0
mux_pow_dyn = 0
mux_area = 0

# Data Memory value dictionary
dataMem_lat_dict = {'256' : 1,
                    '512' : 1,
                    '1024': 1,
                    '2048': 1,
		    '4096':1}

dataMem_pow_dyn_dict = {'256' : 0.16,
                        '512' : 0.24,
                        '1024': 0.33,
                        '2048': 0.57,
			'4096': 0.57}

dataMem_pow_leak_dict = {'256' : 0.044,
                         '512' : 0.078,
                         '1024': 0.147,
                         '2048': 0.33,
			 '4096': 0.33}

dataMem_area_dict = {'256' : 0.00056,
                     '512' : 0.00108,
                     '1024': 0.00192,
                     '2048': 0.00392,
		     '4096': 0.00392}

dataMem_lat_dict = {'256' : 1,
                    '512' : 1,
                    '1024': 1,
                    '2048': 1,
		    '4096':1}

dataMem_pow_dyn_dict = {'256' : 0.16,
                        '512' : 0.24,
                        '1024': 0.33,
                        '2048': 0.57,
			'4096': 0.57}

dataMem_pow_leak_dict = {'256' : 0.044,
                         '512' : 0.078,
                         '1024': 0.147,
                         '2048': 0.33,
			 '4096': 0.33}

dataMem_area_dict = {'256' : 0.00056,
                     '512' : 0.00108,
                     '1024': 0.00192,
                     '2048': 0.00392,
		     '4096': 0.00392}

# Instruction Memory value dictionary
instrnMem_lat_dict = {'512' : 1,
                      '1024': 1,
                      '2048': 1,
		      '4096': 1,
		      '8192': 1,
                      '16384': 1,
                      '32768': 1,
                      '65536': 1,
                      '131072': 1}

instrnMem_pow_dyn_dict = {'512' : 0.46,
                          '1024': 0.53,
                          '2048': 0.65,
		      	  '4096': 0.65,
		          '8192': 0.65,
                          '16384': 0.65,
                          '32768': 0.65,
                          '65536': 0.65,
                          '131072': 0.65}

instrnMem_pow_leak_dict = {'512' : 0.078,
                           '1024': 0.147,
                           '2048': 0.33,
		           '4096': 0.33,
		           '8192': 0.33,
                           '16384': 0.33,
                           '32768': 0.33,
                           '65536': 0.33,
                           '131072': 0.33}


instrnMem_area_dict = {'512' : 0.00108,
                       '1024': 0.00192,
                       '2048': 0.0041,
		       '4096': 0.0041,
		       '8192': 0.0041,
                       '16384': 0.0041,
                       '32768': 0.0041,
                       '65536': 0.0041,
                       '131072': 0.0041}


# Xbar_inMem value dictionary (1 access means reading (dac_res) bits for each xbar row)
# for computing average power of ima - scale dyn_pow down by xbar_size
xbar_inMem_lat_dict = {'32'  : 1, # indexed with xbar size
                       '64'  : 1,
                       '128' : 1,
                       '256' : 1}

xbar_inMem_pow_dyn_read_dict = {'32'  : 0.3,
                                '64'  : 0.7,
                                '128' : 1.7,
                                '256' : 4.7}

xbar_inMem_pow_dyn_write_dict = {'32'  : 0.1,
                                 '64'  : 0.1,
                                 '128' : 0.16,
                                 '256' : 0.2}

xbar_inMem_pow_leak_dict = {'32'  : 0.009,
                            '64'  : 0.02,
                            '128' : 0.04,
                            '256' : 0.075}

xbar_inMem_area_dict = {'32'  : 0.00015,
                        '64'  : 0.00033,
                        '128' : 0.00078,
                        '256' : 0.0019}

# Xbar_outMem value dictionary
# storing 16-bit value
xbar_outMem_lat_dict = {'32'  : 1, # indexed with xbar size
                       '64'   : 1,
                       '128'  : 1,
                       '256'  : 1}

xbar_outMem_pow_dyn_dict = {'32'  : 0.1,
                           '64'   : 0.1,
                           '128'  : 0.16,
                           '256'  : 0.2}

xbar_outMem_pow_leak_dict = {'32'  : 0.009,
                            '64'   : 0.02,
                            '128'  : 0.04,
                            '256'  : 0.075}

xbar_outMem_area_dict = {'32'  : 0.00015,
                        '64'   : 0.00033,
                        '128'  : 0.00078,
                        '256'  : 0.0019}


# Chosen latency based on config file - only for components whose latency is parameter dependent
#xbar_lat = xbar_lat_dict [str(cfg.xbar_bits)][str(cfg.xbar_size)]
'''
xbar_ip_lat_dict = {'0':0, '90':0, '80':0, '70':0, '60':0, '50':0, '40':0, '30':0, '20':0, '10':0}
if cfg.MVMU_ver == "Analog":
      for key, value in xbar_ip_lat_dict.items():
            xbar_ip_lat_dict[key] = xbar_ip_lat
else:
      xbar_ip_lat_dict = digi_param.Digital_xbar_lat_dict[cfg.MVMU_ver][str(cfg.xbar_size)]

xbar_op_lat = xbar_op_lat
xbar_rd_lat = xbar_rd_lat
xbar_wr_lat = xbar_wr_lat
'''
dac_lat = dac_lat_dict [str(cfg.dac_res)]
#FIXME need to review it I can remove adc_lat property
adc_lat = adc_lat_dict [str(cfg.adc_res)] if cfg.adc_type == 'normal' else diff_adc_lat_dict[str(cfg.adc_res)]
xbar_inMem_lat = xbar_inMem_lat_dict[str(cfg.xbar_size)]
xbar_outMem_lat = xbar_outMem_lat_dict[str(cfg.xbar_size)]
instrnMem_lat =  instrnMem_lat_dict[str(cfg.instrnMem_size)]
dataMem_lat =  dataMem_lat_dict[str(cfg.dataMem_size)]

# Chosen area based on config file - only for components whose area is parameter dependent
if cfg.MVMU_ver == "Analog":
        xbar_area = xbar_area_dict['2'][str(cfg.xbar_size)]
else:
        xbar_area = digi_param.Digital_xbar_area_dict[cfg.MVMU_ver][str(cfg.xbar_size)]
dac_area = dac_area_dict [str(cfg.dac_res)]
adc_area = adc_area_dict [str(cfg.adc_res)] if cfg.adc_type == 'normal' else diff_adc_area_dict[str(cfg.adc_res)]
xbar_inMem_area = xbar_inMem_area_dict[str(cfg.xbar_size)]
xbar_outMem_area = xbar_outMem_area_dict[str(cfg.xbar_size)]
instrnMem_area =  instrnMem_area_dict[str(cfg.instrnMem_size)] * math.sqrt(8) #area scaling for 8 bytes per instruction
dataMem_area =  dataMem_area_dict[str(cfg.dataMem_size)]

# Chosen dyn_power based on config file - only for components whose latency is parameter dependent
#xbar_pow_dyn = xbar_pow_dict [str(cfg.xbar_bits)][str(cfg.xbar_size)]
xbar_ip_pow_dyn_dict = xbar_ip_pow_dict
xbar_op_pow_dyn_dict = xbar_op_pow_dict
xbar_rd_pow_dyn_dict = xbar_rd_pow_dict
xbar_wr_pow_dyn_dict = xbar_wr_pow_dict
dac_pow_dyn = dac_pow_dyn_dict [str(cfg.dac_res)]
adc_pow_dyn = adc_pow_dyn_dict [str(cfg.adc_res)] if cfg.adc_type == 'normal' else diff_adc_pow_dyn_dict[str(cfg.adc_res)]
xbar_inMem_pow_dyn_read = xbar_inMem_pow_dyn_read_dict[str(cfg.xbar_size)]
xbar_inMem_pow_dyn_write = xbar_inMem_pow_dyn_write_dict[str(cfg.xbar_size)]
xbar_outMem_pow_dyn = xbar_outMem_pow_dyn_dict[str(cfg.xbar_size)]
instrnMem_pow_dyn =  instrnMem_pow_dyn_dict[str(cfg.instrnMem_size)] * math.sqrt(8) #area scaling for 8 bytes per instruction
dataMem_pow_dyn =  dataMem_pow_dyn_dict[str(cfg.dataMem_size)]

# Energy
xbar_ip_energy_dict = {}
if cfg.MVMU_ver == "Analog":
        for bits_per_cell in range(1, 9):
                xbar_ip_energy_dict[str(bits_per_cell)] = {}
                for sparsity in range(0, 100, 10):
                        xbar_ip_energy_dict[str(bits_per_cell)][str(sparsity)] = \
                                xbar_ip_lat_dict[str(bits_per_cell)] * xbar_ip_pow_dyn_dict[str(bits_per_cell)]
else:
        xbar_ip_energy_dict = digi_param.Digital_xbar_energy_dict[cfg.MVMU_ver][str(cfg.xbar_size)]

# Chosen leak_power based on config file - only for components whose latency is parameter dependent
if cfg.MVMU_ver == "Analog":
        xbar_pow_leak = 0
else:
        xbar_pow_leak = digi_param.Digital_xbar_pow_leak_dict[str(cfg.xbar_size)]
dac_pow_leak = dac_pow_leak_dict [str(cfg.dac_res)]
adc_pow_leak = adc_pow_leak_dict [str(cfg.adc_res)] if cfg.adc_type == 'normal' else diff_adc_pow_leak_dict[str(cfg.adc_res)]
xbar_inMem_pow_leak = xbar_inMem_pow_leak_dict[str(cfg.xbar_size)]
xbar_outMem_pow_leak = xbar_outMem_pow_leak_dict[str(cfg.xbar_size)]
instrnMem_pow_leak =  instrnMem_pow_leak_dict[str(cfg.instrnMem_size)] * math.sqrt(8) #area scaling for 8 bytes per instruction
dataMem_pow_leak =  dataMem_pow_leak_dict[str(cfg.dataMem_size)]

# Core Control unit (control unit and pipeline registers)
ccu_pow = 1.25*0.2 #0.2 for activvity
ccu_area = 0.00145*2.25 #taken similar as edctrl (scaled by power)

# Added here for simplicity now (***can need modification later***)
# The latency of mem access is dependent on when can the ima find edram bys non-busy
memInterface_lat = infinity # infinite latency, here 10000000

#############################################################################################################
# Tile Hierarchy
    # Number of IMAs
    # EDRAM size
    # Shared Bus width
    # Instruction memory size
    # Receive Buffer size
#############################################################################################################

# Tile component latency/pow/area
# EDRAM value dictionary (counter storage is not coounted)
edram_lat_dict = {'8'   : 2,
                  '64'  : 2, #edram access width is constant = 256 bits
                  '128' : 2,
		  '2048': 2,
                  '8192': 2,
                  '16384': 2}

edram_pow_dyn_dict = {'8'   : 17.2/2,
                      '64'  : 17.2/2, # (0.0172 nJ with 2 cycles access latency)
                      '128' : 25.35/2,
		      '2048': 25.35/2,
                      '8192': 25.35/2,
                      '16384': 25.35/2}

edram_pow_leak_dict = {'8'   : 0.46,
                       '64'  : 0.46,
                       '128' : 0.77,
		       '2048': 0.77,
                       '8192': 0.77,
                       '16384': 0.77}

edram_area_dict = {'8'   : 0.086,
                   '64'  : 0.086,
                   '128' : 0.121,
		   '2048': 0.121,
                   '8192': 0.121,
                   '16384': 0.121}

# Tile Instruction Memory value dictionary
tile_instrnMem_lat_dict = {'512': 1,
                          '1024': 1,
                          '2048': 1,
			  '4096': 1,
                          '8192': 1,
                          '16384': 1,
                          '32768': 1,
                          '65536': 1,
                          '131072': 1}

tile_instrnMem_pow_dyn_dict = {'512' : 0.46,
                               '1024': 0.53,
                               '2048': 0.65,
			       '4096': 0.65,
                               '8192': 0.65,
                               '16384': 0.65,
                               '32768': 0.65,
                               '65536': 0.65,
                               '131072': 0.65}

tile_instrnMem_pow_leak_dict = {'512' : 0.078,
                                '1024': 0.147,
                                '2048': 0.33,
			        '4096': 0.33,
                                '8192': 0.33,
                                '16384': 0.33,
                                '32768': 0.33,
                                '65536': 0.33,
                                '131072': 0.33}


tile_instrnMem_area_dict = {'512' : 0.00108,
                            '1024': 0.00192,
                            '2048': 0.0041,
			    '4096': 0.0041,
                            '8192': 0.0041,
                            '16384': 0.0041,
                            '32768': 0.0041,
                            '65536': 0.0041,
                            '131072': 0.0041}


# counter storage (2048 Byte Scratch RAM - 1 counter entry shared by 256 bits of data (16 neurons))
# area scaling (X8)
counter_buff_lat = 1 * math.sqrt(8)
counter_buff_pow_dyn = 0.65/2 * math.sqrt(8)
counter_buff_pow_leak = 0.33/2 * math.sqrt(8)
counter_buff_area = 0.0041 * math.sqrt(8)

# EDRAM to IMA bus values
edram_bus_lat = 1
edram_bus_pow_dyn = 6/2 #bus width = 384, same as issac (over two cycles)
edram_bus_pow_leak = 1/2 #bus width = 384, same as issac
edram_bus_area = 0.090

# EDRAM controller values
edram_ctrl_lat = 1
edram_ctrl_pow_dyn = 0.475
edram_ctrl_pow_leak = 0.05
edram_ctrl_area = 0.00145

# Receive buffer value dictionary - 16 entries (Need to make this a dictionary)
# Increasing to 64 entries
receive_buffer_lat = 1 * math.sqrt(4)
receive_buffer_pow_dyn = 4.48 * math.sqrt(4) # (0.2*256/16)
receive_buffer_pow_leak = 0.09 * math.sqrt(4)
receive_buffer_area = 0.0022 *math.sqrt(4)


# Chosen latency based on config file - only for components whose latency is parameter dependent
edram_lat = edram_lat_dict[str(cfg.edram_size)]
tile_instrnMem_lat = tile_instrnMem_lat_dict[str(cfg.tile_instrnMem_size)]

# Chosen area based on config file - only for components whose area is parameter dependent
edram_area = edram_area_dict[str(cfg.edram_size)]
tile_instrnMem_area = tile_instrnMem_area_dict[str(cfg.tile_instrnMem_size)] * math.sqrt(8) #area scaling for 8 bytes per instruction

# Chosen dynamic power based on config file - only for components whose dynamic power is parameter dependent
edram_pow_dyn = edram_pow_dyn_dict[str(cfg.edram_size)]
tile_instrnMem_pow_dyn = tile_instrnMem_pow_dyn_dict[str(cfg.tile_instrnMem_size)] * math.sqrt(8) #area scaling for 8 bytes per instruction

# Chosen leakage power based on config file - only for components whose leakage power is parameter dependent
edram_pow_leak = edram_pow_leak_dict[str(cfg.edram_size)]
tile_instrnMem_pow_leak = tile_instrnMem_pow_leak_dict[str(cfg.tile_instrnMem_size)] * math.sqrt(8) #area scaling for 8 bytes per instruction

# Tile Control unit
tcu_pow = 0.25*0.2
tcu_area = 0.00145 #taken similar as edctrl

#############################################################################################################
# Node Hierarchy
    # Number of Tiles
    # NOC - Topology (Currently assumes a cmesh (c=4, same as ISSAC))
        # n = number of dimension\
        # k = number of tiles in each dimension
        # c = concentartion (tiles/router)
        # average injection rate (0.25 - a tile injects a new packet for each destination in every four cycles)
#############################################################################################################

# NOC latency dictionary (in terms of flit cycle)
# Note - if inj_rate (packet injection -1 packet - 16 neurons) exceeds 0.025 - there's a problem, NoC needs to be redesigned else network latency will be killing!
# Hence, not provided for
noc_inj_rate_max = 0.025
noc_lat_dict = {'0.001': 29,
                '0.005': 31,
                '0.01' : 34,
                '0.02' : 54,
                '0.025': 115}

noc_area_dict = {'4': 0.047,
                 '8': 0.116}

# Router dynamic power - NOC will be used only if atleast oen of send_queue in node is non_empty
noc_pow_dyn_dict = {'4': 16.13,
                      '8': 51.48}

# Router leakage power - NOC will be used only if atleast oen of send_queue in node is non_empty
noc_pow_leak_dict = {'4': 0.41,
                       '8': 1.04}

# Enter component latency (Based on teh above NOC topological parameters)
# Inter-node Noc (router & channel)
assert (cfg.noc_inj_rate <= noc_inj_rate_max), 'Oops: reconsider NOC design and or DNN mapping, with this inj_rate, NOC data transfer throughput \
will be terrible!'

noc_intra_lat = noc_lat_dict[str(cfg.noc_inj_rate)]
noc_intra_pow_dyn = noc_pow_dyn_dict[str(cfg.noc_num_port)] # per router
noc_intra_pow_leak = noc_pow_leak_dict[str(cfg.noc_num_port)]# per router
noc_intra_area = noc_area_dict[str(cfg.noc_num_port)] # per router

# Hypertransport network (HT)
# Note HT is external to a node, but we consider all tiles in one
# virtual node itself for simplicity
# HT numbers from ISAAC = 6.4GB/s = 6.4B/ ns = 1packet(16*2 Bytes) = 5ns
ht_lat = 5 #latency per packet
noc_inter_lat = ht_lat + noc_intra_lat #navigate to the node, then to tile within node
noc_inter_pow_dyn = 10400 #10.4W
noc_inter_pow_leak = 0
noc_inter_area = 22.88

