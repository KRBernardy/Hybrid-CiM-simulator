# This file contains the configurable parameters in DPE (all hierarchies - IMA, Tile, Node)
## All user specified parameters are provided by this file only


from data_config import datacfg

## Debug - 0 (1): dpe simulation will (won't) produce ima/tile traces while simulating
cycles_max = 100000000 # Put both these to very large numbers (when design is bug-free)!
debug = 1
xbar_record = 1
inference = 1
training = not(inference)
sparse_opt = 1 # Flag for Sparsity optimisaton (Make it 0 for only dense computations)
accuracy_only = 0 # Flag for only accuracy computation. Will not provide any energy numbers

## Variable to define the type of MVMU
# One of "Analog", "Digital_V1" or "Digital_V2" 
# Digital_V1 has compressed inputs (Data+Offset style)
# Digital_V2 has uncompressed inputs (Skips computations for 0 activation)
MVMU_ver = "Analog"

## IMA configurable parameters (permissible values for each parameter provided here)
## Instruction generation - affected by xbar_bits, num_xbar, xbar_size.
# xbar_bits: 2, 4, 6
# num_xbar: positive integercd
# xbar_size: 32, 64, 128, 256
# dac_res: positive integer <= num_bits
# adc_res: positive integer <= num_bits
# num_adc: positive integer <= num_xbar (doesn't allow more than one ADC per xbar)
# num_ALU: positive integer
# dataMem_size: (in Bytes) - 256, 512, 1024, 2048 (affects instrn width, hence capped)
# instrnMem_size: (in Bytes) - 512, 1024, 2048

# Fixed parameters
addr_width = 32 # Added to address larger address space for conv layers (#TODO: Compiler needs to fix shared memory reuse)
data_width = datacfg.num_bits # (in bits)

xbdata_width = data_width # (in bits)
instrn_width = 48 # (in bits)
# Input and Weight parameters
input_prec = datacfg.num_bits
weight_width = datacfg.num_bits
# Change here - Specify the IMA parameters here
num_matrix = 6 # each matrix is 1-fw logical xbar for inference and 1-fw, 1-bw, and 1 delta logical xbar for training. Each logical xbar for inference is 8-fw physical xbar and for training  8-fw, 8-bw and 16-delta physical xbars.
xbar_size = 128
dac_res = 1
# ADC configuration
adc_res = 9 # this value should be around 4 to 16.
adc_type = 'normal' # 'normal' or 'differential'
# num_column_per_adc is number of xbar columns per adc. Notice that a positive column next to a negative column is considered as one column here.
# Needs to make sure that xbar_size is multiple of num_column_per_adc. Recommanded value is 16 for normal ADC and 8 for differential ADC ().
num_column_per_adc = 16

num_mux_per_xbar = xbar_size // num_column_per_adc * 2 # number of MUXs per xbar, for both positive and negative
num_mux = num_matrix * 2 * datacfg.ReRAM_xbar_num * num_mux_per_xbar # number of MUXs in total

num_adc_per_xbar = num_mux_per_xbar if adc_type == 'normal' else num_mux_per_xbar / 2 # differential ADC has half the number of muxes
num_adc = num_matrix * 2 * datacfg.ReRAM_xbar_num * num_adc_per_xbar # number of ADCs in total

# The idea is to have different ADC resolution value for each ADC.
# The number of ADC if defined by num_adc property. Currently it is 2 * num_matrix(2) = 4
# NOTE: Only taking in account indexes 0 and 2, 1 and 3 are ignored, because ADCs 1 and 3 are assumed t be equal to 0 and 2. 
adc_res_new = {
                'matrix_adc_0' : 8,
                'matrix_adc_1' : 8,
                'matrix_adc_2' : 8,
                'matrix_adc_3' : 8
              }

num_ALU = num_matrix*2
#dataMem_size = num_matrix*(6*xbar_size) # 4 for 4 input spaces within matrix (1 for f/b each, 2 for d)
dataMem_size = 4096 # same as in compiler
instrnMem_size = 131072 #in entries

# This depends on above parameters
if (training):
    datamem_off = xbar_size * (num_matrix*6) # each matrix has 6 memory spaces (1 for f/b, 2 for d)

if (inference):
    datamem_off = xbar_size * (num_matrix*2) # each matrix has 2 memory spaces ( 1 input Xbar memory and 1 output Xbar memory) 

#phy2log_ratio = num_bits / xbar_bits # ratio of physical to logical xbar #vaulue is 8
lr = 0.25 # learning rate for updates to d-xbar

## Tile configurable parameters (permissible values for each parameter provided here)
## Instruction generation - affected by num_ima
# num_ima: positive integer
# edram buswidth: positive integer <= 16 (actual buswidth - this integer*data_width)
# edram_size: (in KiloBytes) - 64, 128, 256, 512
# receive_buffer_depth: 4, 8, 12, 16, 32 (number of edram buffer entries (each entry maps to a virtual tile)) \
#        puts a cap on the maximum num ber of tiles that can send data to a tile in next layer
# receive_buffer_width: edram_buswidth/data_width (Fixed - in terms of number of neurons)
# tile_instrnMem_size: 256, 512, 1024 (in Bytes)

# Fixed parameters
instrn_width = 48 # bits (op-2, vtile_id-6, send/receive_width-8, target_addr/counter-16, vw-8, mem_addr-16)
edram_buswidth = 256 # in bits
#receive_buffer_depth = 16
receive_buffer_depth = 150 #set equal to num_tile_max
receive_buffer_width =  edram_buswidth // datacfg.num_bits # size of receive buffeer entry (in terms of number of neurons)

# Change here - Specify the Tile parameters here
num_ima = 8
edram_size = 16384 # in Kilobytes (64 KB - same as issac)
tile_instrnMem_size = 131072 # in entries

## Node configurable parameters (permissible values for each parameter provided here)
## Instruction generation - affected by num_tile
# num_tile_compute =  positive integer
# inj_rate < 0.2 (depends on the mapping)
# num_port: 4, 8

# Fixed parameters
# NOC topology: cmesh (n=2, k=4, c=4) - can fit k*n*c tiles
cmesh_c = 4
num_bits_tileId =32
flit_width = 32
packet_width = edram_buswidth/data_width #in multiples of flits (data considered only - booksim consider address itself)
# (b bit of address = logN, N is the number of nodes)

# Change here - Specify the Node parameters here
num_tile_compute = 3
num_tile_max = 168.0 # maximum number of tiles per node
num_inj_max = num_tile_max # [conservative] max number of packet injections that can occur in a cycle (each tile injects a packet into NOC each cycle)
noc_inj_rate = 0.005
noc_num_port = 4

## Node parameters - Our way of simulation just assumes all tile in one actual node
num_node = 1

# Do not change this - total number of tiles
num_tile = num_node * num_tile_compute + 2 # +1 for first tile (I/O tile) - dummy, others - compute

#Security parameters - Used to verify if the model used is encryted or authenticated (set by dpe.py)
#Do not change
encrypted = False
authenticated = False
cypher_name = ''
cypher_hash = ''

# Thread number for parallel processing
thread_num = 8
