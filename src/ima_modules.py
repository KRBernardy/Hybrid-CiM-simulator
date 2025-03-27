# Defining all modules within an ima and its methods (Propagate and or Update)
# Also define sime support functions (commonly used)
# 'Propagate' is the combinational part evaluation
# 'Update' - (NOT BEING USED CURRENTLY!) is the flip fop evaluation (for clocked elements of the circuit only)

import sys

import numpy as np
import include.constants as param
import include.config as cfg
from include.data_config import datacfg, max_val, min_val
import math
from data_convert import *

import line_profiler

class xbar (object):
    def __init__ (self, xbar_size, bits_per_cell = 2, xbar_value= 'nil' ):
        # define num_accesses for different operations
        # parallel reads (inner-product)
        self.num_access = { '0':0, '90': 0,'80': 0,'70': 0,'60': 0,'50': 0,'40': 0,'30': 0,'20': 0,'10': 0}

        self.num_access_rd = 0 # serial reads
        self.num_access_wr = 0 # serial writes

        # define latency for various xbar operations
        # latency for write should vary with bits_per_cell, to be add later
        self.latency_ip = param.xbar_ip_lat_dict[str(bits_per_cell)]
        self.latency_op = param.xbar_op_lat_dict[str(bits_per_cell)]
        self.latency_rd = param.xbar_rd_lat_dict[str(bits_per_cell)]
        self.latency_wr = param.xbar_wr_lat_dict[str(bits_per_cell)]

        # xbar_value is the weights meant for one crossbar
        self.xbar_size = xbar_size
        #self.xbar_value = np.random.randn(xbar_size, xbar_size)
        self.xbar_value_pos = np.zeros((xbar_size, xbar_size))
        self.xbar_value_neg = np.zeros((xbar_size, xbar_size))
        # unprogrammed xbar contains zeros
        if (xbar_value != 'nil'):
            for i in range(xbar_size):
                for j in range(xbar_size):
                    if xbar_value[i][j] > 0:
                        self.xbar_value_pos[i][j] = xbar_value[i][j]
                    else:
                        self.xbar_value_neg[i][j] = -1 * xbar_value[i][j]

        # xbar output currents are recorded fro analysis of applicable
        self.xb_record = []
        self.bits_per_cell = bits_per_cell


    # Records the xbar currents
    def record (self, xb_out):
        self.xb_record.append(xb_out)

    def bitsPerCell(self):
        return str(self.bits_per_cell)

    def get_value(self, accurate = False):
        # This function is used to get the xbar value for quick calculation
        if accurate:
            return [self.xbar_value_pos, self.xbar_value_neg]
        noise_pos = np.random.normal(0, param.ReRAM_read_sigma, (self.xbar_size, self.xbar_size))
        value_with_noise_pos = self.xbar_value_pos + noise_pos
        value_with_noise_pos[value_with_noise_pos < 0] = 0 # conductance should not be negative, set all negataive values 0

        noise_neg = np.random.normal(0, param.ReRAM_read_sigma, (self.xbar_size, self.xbar_size))
        value_with_noise_neg = self.xbar_value_neg + noise_neg
        value_with_noise_neg[value_with_noise_neg < 0] = 0 # conductance should not be negative, set all negataive values 0
        return [value_with_noise_pos, value_with_noise_neg]

    # programs the entire xbar during configuration phase
    def program (self, xbar_value = ''):
        # programs the crossbar with given matrix values
        val_size = np.shape (xbar_value)
        size_max = cfg.xbar_size
        assert (val_size[0] <= size_max and val_size[1] <= size_max), \
                    'Xbar values format should be a numpy array of the xbar dimensions'
        #self.xbar_value[0:val_size[0], 0:val_size[1]] = xbar_value.copy ()
        for i in range(self.xbar_size):
                for j in range(self.xbar_size):
                    if xbar_value[i][j] > 0:
                        self.xbar_value_pos[i][j] = xbar_value[i][j]
                        self.xbar_value_neg[i][j] = 0
                    else:
                        self.xbar_value_neg[i][j] = -1 * xbar_value[i][j]
                        self.xbar_value_pos[i][j] = 0

    # writes to a location on xbar
    def write (self, k, l, value):
        assert (k < cfg.xbar_size), 'row entry exceeds xbar size'
        assert (l < cfg.xbar_size), 'col entry exceeds xbar size'
        assert (type(value) == float), 'value written to xbar should be float'
        self.num_access_wr += 1
        if (value > 0):
            self.xbar_value_pos[k][l] = value
            self.xbar_value_neg[k][l] = 0
        else:
            self.xbar_value_pos[k][l] = 0
            self.xbar_value_neg[k][l] = -1 * value

    # reads a location on xbar with ReRAM accuracy degradtion
    def read (self, k, l, accurate = False):
        assert (k < cfg.xbar_size), 'row entry exceeds xbar size'
        assert (l < cfg.xbar_size), 'col entry exceeds xbar size'
        self.num_access_rd += 1
        if self.xbar_value_pos[k][l] > 0:
            value = self.xbar_value_pos[k][l]
        else:
            value = -1 * self.xbar_value_neg[k][l]
        if (accurate == False):
            return np.random.normal(value, param.ReRAM_read_sigma)
        else:
            return value
    

    def getIpLatency (self):
        return self.latency_ip

    def getOpLatency (self):
        return self.latency_op

    def getRdLatency (self):
        return self.latency_rd

    def getWrLatency (self):
        return self.latency_wr

    #input here should be float list, we use float to represent analog values
    def propagate (self, inp = 'nil', sparsity = 0, accurate = False):
        if cfg.MVMU_ver == "Analog":
            self.num_access['0'] += 1
        else:
            self.num_access[str(sparsity)] +=1
        assert (inp != 'nil'), 'propagate needs a non-nil input'
        assert (len(inp) == self.xbar_size), 'xbar input size mismatch'
        # add noise when reading if didn't ask for acccurate value
        if accurate:
            out_pos = np.dot(inp, self.xbar_value_pos)
            out_neg = np.dot(inp, self.xbar_value_neg)
        else:
            rng = np.random.default_rng()
            noise_pos = rng.normal(0, param.ReRAM_read_sigma, (self.xbar_size, self.xbar_size))
            value_with_noise_pos = self.xbar_value_pos + noise_pos
            value_with_noise_pos[value_with_noise_pos < 0] = 0 # conductance should not be negative, set all negataive values 0
            out_pos = np.dot(inp, value_with_noise_pos.transpose())

            noise_neg = rng.normal(0, param.ReRAM_read_sigma, (self.xbar_size, self.xbar_size))
            value_with_noise_neg = self.xbar_value_neg + noise_neg
            value_with_noise_neg[value_with_noise_neg < 0] = 0 # conductance should not be negative, set all negataive values 0
            out_neg = np.dot(inp, value_with_noise_neg.transpose())
        self.record([out_pos, out_neg])
        return [out_pos, out_neg]

    def propagate_dummy (self, inp = 'nil', sparsity = 0, accurate = False):
        # data input is list of bit strings (of length dac_res) - fixed point binary
        assert (inp != 'nil'), 'propagate needs a non-nil input'
        assert (len(inp) == self.xbar_size), 'xbar input size mismatch'
        
        #Modification to accomodate sparsity and digital crossbars
        if cfg.MVMU_ver == "Analog":
            self.num_access['0'] += 1
        else:
            self.num_access[str(sparsity)] +=1

        # convert input from fixed point binary (string) to float
        inp_float = [0.0] * self.xbar_size
        for i in range(len(inp)):
            # extend data to num_bits for computation (sign extended)
            temp_inp = (cfg.num_bits - cfg.dac_res) * '0' + inp[i]
            inp_float[i] = fixed2float(temp_inp, datacfg.int_bits, datacfg.frac_bits)
        inp_float = np.asarray (inp_float)
        # add noise when reading if didn't ask for acccurate value
        if accurate:
            out_float = np.dot(inp, self.xbar_value)
        else:
            noise = np.random.normal(0, param.ReRAM_read_sigma, (self.xbar_size, self.xbar_size))
            out_float = np.dot(inp, self.xbar_value + noise)

        # record xbar_i if applicable
        if (cfg.xbar_record):
            self.record(out_float)
        '''
        # convert float back to fixed point binary
        out_fixed  = [''] * self.xbar_size
        for i in range(len(out_fixed)):
            out_fixed[i] = float2fixed(out_float[i], cfg.int_bits, cfg.frac_bits)

        #return out_fixed
        '''
        # we use float to represent analog values
        return out_float

# xbar_op class supports both mvm (inner-product) and vvo (outer-product) operations
# TODO this is used to deal with the delta when training. not modified yet. further investigation on how ReRAM accuracy degradtion needed
class xbar_op (xbar):
    # add function for outer_product computation
    def propagate_op (self, inp1, inp2, lr, in1_bit, in2_bit):
        # inner-product and outer_product functions should have different energies (and other metrics) - NEEDS UPDATE
        self.num_access['0'] += 1
        # check both data inputs
        assert (inp1 != 'nil' and inp2 != 'nil'), 'propagate needs a non-nil inputs'
        assert ((len(inp1) == self.xbar_size) and (len(inp1[0]) == in1_bit)), 'inp1 size mismatch - should be \
            xbar_sized list with each element being a string of dac_res length'
        assert ((len(inp2) == self.xbar_size) and (len(inp2[0]) == in2_bit)), 'inp2 size mismatch - should be \
            xbar_sized list with each element being a string of xbar_bits length'

        # convert inp1 (dac_res bits) and inp2 (xbar_bits) from fixed point binary string to float
        inp1_float = [0.0] * self.xbar_size
        inp2_float = [0.0] * self.xbar_size
        for i in range(self.xbar_size):
            # extend data to num_bits for computation (sign extended)
            temp_inp1 = (cfg.num_bits - in1_bit) * '0' + inp1[i]
            temp_inp2 = (cfg.num_bits - in2_bit) * '0' + inp2[i]
            inp1_float[i] = fixed2float(temp_inp1, datacfg.int_bits, datacfg.frac_bits)
            inp2_float[i] = fixed2float(temp_inp2, datacfg.int_bits, datacfg.frac_bits)
        inp1_float = np.asarray (inp1_float)
        inp2_float = np.asarray (inp2_float)

        # compute outer product and accumulate (scaled by learning rate - lr)
        delta = lr * np.outer (inp1_float, inp2_float)
        self.xbar_value += delta
        # return delta calculated for debug only
        return delta


class dac (object):
    def __init__ (self, dac_res):
        # define num_access
        self.num_access = 0

        # define latency
        self.latency = param.dac_lat

        self.dac_res = dac_res

    def getLatency (self):
        return self.latency

    def bin2real (self, inp, num_bits):
        # gets a n-bit (n = dac_res) digital value & returns an analog voltage value
        inp_max = '1' * num_bits # string with all 1s
        analog_max = param.vdd
        frac = int(inp, 2) / float(int(inp_max, 2))
        return analog_max * frac

    def propagate (self, inp):
        self.num_access += 1
        if (inp == ''):
            inp = '0' * cfg.dac_res
        assert ((type(inp) == str) and (len(inp) == self.dac_res)), 'dac input type/size (bits) mismatch (string expected)'
        num_bits = self.dac_res
        return self.bin2real (inp, num_bits)


# A dac_array is an arrays of DACs private to a xbar
class dac_array (object):
    def __init__ (self, xbar_size, dac_res):
        # define latency
        self.latency = param.dac_lat

        # generate multiple dacs (one per xbar input)
        self.dac_list = []
        self.xbar_size = xbar_size
        for i in range(xbar_size):
            temp_dac = dac (dac_res)
            self.dac_list.append(temp_dac)

    def getLatency (self):
        return self.latency

    def propagate (self, inp_list):
        assert (len(inp_list) == self.xbar_size), 'dac_array input list size mismatch'
        out_list = []
        for i in range(self.xbar_size):
            temp_out = self.dac_list[i].propagate(inp_list[i])
            out_list.append(temp_out)
        return out_list

    # HACK - until propagate doesn't have correct analog functionality
    def propagate_dummy (self, inp_list):
        assert (len(inp_list) == self.xbar_size), 'dac_array input list size mismatch'

        # just to keep track of individual dacs
        junk_list = []
        for i in range(self.xbar_size):
            temp_out = self.dac_list[i].propagate(inp_list[i])
            junk_list.append(temp_out)
        # just to keep track of individual dacs

        out_list = inp_list [:]
        return out_list


# Probably - also doing the sampling part of (sample and hold) inside
class adc (object):
    def __init__ (self, adc_res, bits_per_cell = 2):
        # define num_access
        self.num_access = { 'n':0, 'n/2': 0,'n/4': 0,'n/8': 0,'n/16': 0,'n/32': 0,'n/64': 0,'n/128': 0}
        
        # define latency
        self.latency = param.adc_lat_dict[str(adc_res)]

        self.adc_res = adc_res

        self.bits_per_cell = bits_per_cell
        self.max_val = 2 ** adc_res - 1

        num_levels = 2**adc_res
        conductance_step = (param.xbar_conductance_max - param.xbar_conductance_min) / ((2 ** bits_per_cell) - 1)
        voltage_step = param.vdd / ((2 ** cfg.dac_res) - 1)
        self.current_step = voltage_step * conductance_step

    def getLatency (self):
        self.latency = param.adc_lat_dict[str(self.adc_res)]
        return self.latency

    def real2bin (self, inp, num_bits, bits_per_cell = 2, dac_res = cfg.dac_res, return_type = 'bin'):
        int_value = min(int(inp / self.current_step), max_val) # clip to max_val
        if return_type == 'int':
            return int_value
        bin_value = bin(int_value)[2:]
        return ('0'*(num_bits - len(bin_value)) + bin_value)

    # Here we allow the adc to deal with negative inputs
    # in real circult it should calculate twice, first time for all positive value
    def propagate (self, inp, sparsity = 0, return_type = 'bin'):
        assert (type(inp) in [float, np.float32, np.float64]), 'adc input type mismatch (float, np.float32, np.float64 expected)'
        assert (return_type in ['bin', 'int']), 'return_type should be bin or int'
        if sparsity > 0:
            reduction_level = min(7, int(np.log2(100/(100-sparsity))))
            self.adc_res = max(1, cfg.adc_res - reduction_level)
            self.num_access[f'n/{2**reduction_level}'] += 1
        else:
            self.adc_res = cfg.adc_res
            self.num_access['n'] += 1
        
        int_value = min(int(inp / self.current_step), self.max_val) # clip to max_val

        if return_type == 'int':
            return int_value
        return int2bin(int_value, self.adc_res)

    # HACK - until propagate doesn't have correct analog functionality
    def propagate_dummy (self, inp, sparsity = 0):
        if sparsity<50:
            self.num_access['n'] += 1
            self.adc_res = cfg.adc_res
        elif sparsity<75:
            self.num_access['n/2'] += 1
            self.adc_res = cfg.adc_res-1
        elif sparsity<87.5:
            self.num_access['n/4'] += 1
            self.adc_res = cfg.adc_res-2
        elif sparsity<93.75:
            self.num_access['n/8'] += 1
            self.adc_res = cfg.adc_res-3
        elif sparsity<96.875:
            self.num_access['n/16'] += 1
            self.adc_res = cfg.adc_res-4
        elif sparsity<98.4375:
            self.num_access['n/32'] += 1
            self.adc_res = cfg.adc_res-5
        elif sparsity<99.21875:
            self.num_access['n/64'] += 1
            self.adc_res = cfg.adc_res-6
        else:
            self.num_access['n/128'] += 1
            self.adc_res = cfg.adc_res-7
        if(self.adc_res<=0):
            self.adc_res = 1

        return inp
    
class differential_adc (object):
    def __init__ (self, adc_res, bits_per_cell = 2):
        # define num_access
        self.num_access = { 'n':0, 'n/2': 0,'n/4': 0,'n/8': 0,'n/16': 0,'n/32': 0,'n/64': 0,'n/128': 0}
        
        # define latency
        self.latency = param.diff_adc_lat_dict[str(adc_res)]

        self.adc_res = adc_res

        self.bits_per_cell = bits_per_cell
        self.max_val = 2 ** adc_res - 1
        self.min_val = -1 * 2 ** adc_res

        num_levels = 2**adc_res
        conductance_step = (param.xbar_conductance_max - param.xbar_conductance_min) / ((2 ** bits_per_cell) - 1)
        voltage_step = param.vdd / ((2 ** cfg.dac_res) - 1)
        self.current_step = voltage_step * conductance_step

    def getLatency (self):
        self.latency = param.diff_adc_lat_dict[str(self.adc_res)]
        return self.latency

    def real2bin (self, inp_pos, inp_neg, num_bits, bits_per_cell = 2, dac_res = cfg.dac_res, return_type = 'bin'):
        num_levels = 2**num_bits
        conductance_step = (param.xbar_conductance_max - param.xbar_conductance_min) / ((2 ** bits_per_cell) - 1)
        voltage_step = param.vdd / ((2 ** dac_res) - 1)
        current_step = voltage_step * conductance_step
        int_value = int(float(inp_pos - inp_neg) / float(current_step))
        try:
            assert(abs(int_value) < num_levels)
        except AssertionError:
            print("ADC overflow")
            print(("int_value: ", int_value))
            print(("num_levels: ", num_levels))
            sys.exit(1)
        if return_type == 'int':
            return int_value
        return int2bin(int_value, num_bits)

    # Here we allow the adc to deal with negative inputs
    # in real circult it should calculate twice, first time for all positive value
    def propagate (self, inp_pos, inp_neg, sparsity = 0, return_type = 'bin'):
        assert (type(inp_pos), type(inp_neg) in [float, np.float32, np.float64]), 'adc input type mismatch (float, np.float32, np.float64 expected)'
        assert (return_type in ['bin', 'int']), 'return_type should be bin or int'
        if sparsity > 0:
            reduction_level = min(7, int(np.log2(100/(100-sparsity))))
            self.adc_res = max(1, cfg.adc_res - reduction_level)
            self.num_access[f'n/{2**reduction_level}'] += 1
        else:
            self.adc_res = cfg.adc_res
            self.num_access['n'] += 1
        
        int_value = int((inp_pos - inp_neg) / self.current_step)
        int_value = min(int_value, max_val) # clip to max_val
        int_value = max(int_value, min_val) # clip to min_val

        if return_type == 'int':
            return int_value
        return int2bin(int_value, self.adc_res)

# Doesn't replicate the exact (sample and hold) functionality (just does hold)
class sampleNhold (object):
    def __init__ (self, xbar_size):
        # define num_access
        self.num_access = 0

        # define latency
        self.latency = param.snh_lat

        self.hold_latch = np.zeros(xbar_size)

    def getLatency (self):
        return self.latency

    # propagate is sample
    def propagate (self, inp_list):
        self.num_access += 1
        assert (type(inp_list[0]) in [float, np.float32, np.float64]), 'sample&hold input should be analog (float)'
        assert (len(inp_list) == len(self.hold_latch)), 'sample&hold input size mismatch'
        self.hold_latch = inp_list
    
    def read(self):
        return self.hold_latch

    def propagate_dummy (self, inp_list):
        self.num_access += 1
        assert (len(inp_list) == cfg.xbar_size), 'sample&hold input size mismatch'
        out_list = inp_list[:]
        return out_list

# Note the mux instantiations will be analog mux
class mux (object):
    def __init__ (self, num_in):
        # define num_access
        self.num_access = 0

        # define latency
        self.latency = param.mux_lat

        # num_in is the inputs for the multiplexer
        self.num_in = num_in

    def getLatency (self):
        return self.latency

    def propagate (self, inp_list, sel):
        self.num_access += 1
        assert (len(inp_list) == self.num_in), 'Mux input list size mismatch'
        assert ((type(sel) == int) & (-1 < sel < self.num_in)), 'Mux select input size/type error'
        return inp_list[sel]

    # Note for all practical purpose we will use prop_dummy (mux funtinality is taken care of in code)
    def propagate_dummy (self):
        self.num_access += 1


#### Needs some change - add function op (for instance, shift bits for shift)
## Needs to add ALU overflow check/mitigation
class alu (object):
    def __init__ (self):
        self.num_access_div = 0
        self.num_access_mul = 0
        self.num_access_act = 0
        self.num_access_sna = 0
        self.num_access_sns = 0
        self.num_access_other = 0

        # define latency
        self.latency = param.alu_lat

        # Arithmetic operations
        def add (a, b):
            self.num_access_other += 1
            return (a + b)
        def sub (a, b):
            self.num_access_other += 1
            return (a - b)
        def shift_add (a, b):
            self.num_access_sna += 1
            return (a + b) # does add, b is already shifted
        def shift_sub (a, b):
            self.num_access_sns += 1
            return (a - b) # does subtract, b is already shifted
        def multiply (a, b):
            self.num_access_mul += 1
            return (a * b)

        # Neuronal operations - put here for simplicity (can be made a separate unit)
        # Using aluop (arith./neuronal) dependent power numbers (they will be separate units in harwdare)
        def sigmoid (a, b): # b is unused
            self.num_access_act += 1
            return 1 / (1 + math.exp(-a))
        def tanh (a, b): # b is unused
            self.num_access_act += 1
            return np.tanh(a)
        def relu (a, b): # b is unused
            self.num_access_other += 1
            out = a if (a > 0) else 0
            return out

        # for max-pool layer
        def max_val (a,b):
            self.num_access_other += 1
            return max (a,b)

        self.options = {'add':add, 'sub':sub, 'sna':shift_add, 'sns':shift_sub, 'mul':multiply,\
                'sig':sigmoid, 'tanh':tanh, 'relu':relu, 'max': max_val}

    def getLatency (self):
        return self.latency

    def propagate (self, a, b, aluop, c = 0, return_type = 'fixed'): # c can be shift operand for sna operation (add others later)
        assert ((type(aluop) == str) and (aluop in list(self.options.keys()))), 'Invalid alu_op'
        assert (type(c) == int or (type(c) == str and len(c) == datacfg.num_bits)), 'ALU sna: shift = int/ num_bit str'
        assert (return_type == 'fixed' or return_type == 'float'), "return_type can only be 'fixed' or 'float'"
        if (type(c) == str):
            c = bin2int (c, datacfg.num_bits)
        if type(a) == str:
            a = fixed2float (a, datacfg.int_bits, datacfg.frac_bits)
        if (aluop == 'sna' or aluop == 'sns'): # shift left in fixed point binary
            if type(b) == str:
                if (b == ''):
                    b = 0
                else:
                    if c >= 0:
                        b = b + '0' * c
                        b = fixed2float (b, datacfg.int_bits, datacfg.frac_bits)
                    else:
                        b = b[:c]
                        b = fixed2float (b, datacfg.int_bits, datacfg.frac_bits)
            else:
                b = b * (2 ** c)
        else:
            if type(b) == str: # shift left in fixed point binary
                if b == '':
                    b = 0
                else:
                    b = fixed2float (b, datacfg.int_bits, datacfg.frac_bits)
        out = self.options[aluop] (a, b)
        # overflow needs to be detected while conversion
        out = min(max_val, out) # clip to max_val
        ovf = 0
        if (return_type == 'fixed'):
            out = float2fixed (out, datacfg.int_bits, datacfg.frac_bits)
        return [out, ovf]

    # for functionality define a propagate float for use in inter-xbar shift-and-adds
    # Note: xbar_propagate uses np.dot to compute dot product in float
    # here float to fixed conversion of xbar output (and subsequent alu.propagate) cannot be used
    # unless np.dot is implement using fixed point computation
    def propagate_float (self, a, b, aluop, c=0):
        assert ((type(aluop) == str) and (aluop in list(self.options.keys()))), 'Invalid alu_op'
        assert (type(c) == int or (type(c) == str and len(c) == cfg.num_bits)), 'ALU sna: shift = int/ num_bit str'
        if (type(c) == str):
            c = bin2int (c, cfg.num_bits)
        a = fixed2float (a, datacfg.int_bits, datacfg.frac_bits)
        if (b == ''):
            b = 0
        else:
            if (aluop == 'sna' or aluop == 'sns'): # shift left in fixed point binary
                b = b + '0' * c
            b = fixed2float (b, datacfg.int_bits, datacfg.frac_bits)
        out = self.options[aluop] (a, b)
        # overflow needs to be detected while conversion
        ovf = 0
        return [out, ovf]


# Integer ALU
class alu_int (object):
    def __init__ (self):
        self.num_access_div = 0
        self.num_access_mul = 0
        self.num_access_other = 0

        # define latency
        self.latency = param.alu_lat

        # Arithmetic operations
        def add (a, b):
            self.num_access_other += 1
            return (a + b)
        def sub (a, b):
            self.num_access_other += 1
            return (a - b)
        def multiply (a, b):
            self.num_access_mul += 1
            return (a * b)
        def divide (a, b):
            self.num_access_div += 1
            return int((a/b))
        def mod (a, b):
            self.num_access_other += 1
            return int((a%b))
        def eq_chk (a,b):
            self.num_access_other += 1
            return (a == b)

        self.options = {'add':add, 'sub':sub, 'mul':multiply, 'div':divide, 'mod':mod, 'eq_chk':eq_chk}

    def getLatency (self):
        return self.latency

    def propagate (self, a, b, aluop):
        assert ((type(aluop) == str) and (aluop in list(self.options.keys()))), 'Invalid alu_op'
        a = bin2int (a, cfg.num_bits)
        b = bin2int (b, cfg.num_bits)
        out = self.options[aluop] (a, b)
        # overflow needs to be detected while conversion
        ovf = 0
        out = int2bin(out, cfg.num_bits)
        return [out, ovf]


# Assumes a half-word oriented memory (each entry - 16 bits)
class memory (object):
    def __init__ (self, size, addr_offset = 0):
        # define num_access
        self.num_access = 0

        # define latency
        self.latency = param.dataMem_lat

        # memfile will store half-word (16 bits digital data) length strings
        self.size = size
        self.memfile = [''] * size

        self.addr_start = addr_offset
        self.addr_end = self.addr_start + self.size -1

    def getLatency (self):
        return self.latency

    def read (self, addr):
        self.num_access += 1
        assert (type(addr) == int), 'addr type should be int'
        assert (self.addr_start <= addr <= self.addr_end), 'addr exceeds the memory bounds'
        return self.memfile[addr - self.addr_start]


    def write (self, addr, data, type_t='data'):
        self.num_access += 1
        assert (type(addr) == int), 'addr type should be int'
        assert (self.addr_start <= addr <= self.addr_end), 'addr exceeds the memory bounds'
        #print 'length of data ' + str(len(data))
        #assert ((type(data) ==  str) and (len(data) == cfg.data_width)), 'data should be a string with mem_width bits'
        assert ((type(data) == str) and ((type_t == 'data') or (type_t == 'addr'))) # UPDATE - Pointer/address for LD/ST written by previous SET instrn. can be larger than data_width
        if (type_t == 'data'):
            assert (len(data) == cfg.data_width)
        else:
            assert (len(data) == cfg.addr_width) # Specification for pointer (or addres type data)
        self.memfile[addr - self.addr_start] = data
            

    def reset (self):
        self.num_access += 1
        self.memfile = [''] * self.size


# xbar input memory reads differently than typical memory
# Each read is a shift and read operation
class xb_inMem (object):
    def __init__ (self, xbar_size):
        # define num_access
        self.num_access_read = 0
        self.num_access_write = 0

        # define latency
        self.latency = param.xbar_inMem_lat

        # size equals the xbar_size, each entry being to
        self.xbar_size = xbar_size
        self.memfile = [''] * self.xbar_size

    def getLatency (self):
        return self.latency

    # reads an entry (typical memory read)
    def read_n (self, addr):
        return self.memfile[addr]

    # reads & shifts all entries in parallel
    def read (self, num_bits):
        self.num_access_read += 1
        out_list = []
        for i in range(self.xbar_size):
            value = self.memfile[i]
            #self.memfile[i] = '0'*num_bits + value[:-1*num_bits]
            self.memfile[i] = value[-1*num_bits:] + value[:-1*num_bits]
            out_list.append(value[-1*num_bits:])
        return out_list

    def read_all (self):
        out_list = []
        for i in range(self.xbar_size):
            value = self.memfile[i]
            if value == '':
                value = '0' * cfg.data_width
            out_list.append(fixed2float(value, datacfg.int_bits, datacfg.frac_bits))
        return out_list

    def write (self, addr, data):
        self.num_access_write += 1
        assert (type(addr) == int), 'addr type should be int'
        assert (-1 < addr < self.xbar_size), 'addr exceeds the memory bounds'
        assert ((type(data) ==  str) and (len(data) == cfg.xbdata_width)), 'data should be a string with xbdata_width bits'
        self.memfile[addr] = data

    def reset (self):
        self.num_access += 1
        self.memfile = [''] * self.xbar_size

    # Updated - (input arrangement - depth major -> row -> column)
    # output computation - row wise
    def stride (self, val1, val2): #val1 and val2 come from r1 and r2 and will bint by default
        assert (type(val1) == int and type(val2) == int), 'stride: check data type of val1 and val2'
        if (val1 > 0 and val2 > 0): #val1 and  val2 both zero means hw support for stride in dpe isn't being used
            # Needs a separate access - stride is different than read/write
            self.num_access += 1
            temp_memfile = [''] * self.xbar_size
            for i in range (int(math.ceil(self.xbar_size / val2))):
                for j in range (val2-val1):
                    temp_memfile[i*val2+j] = self.memfile[i*val2 + j+val1]
                    #print ('from src ', self.memfile[i*val2 + j+val1], 'to dest', temp_memfile[i*val2+j])
            self.memfile = temp_memfile [:]


# xbar output memory
class xb_outMem (xb_inMem):
    def __init__ (self, xbar_size):
        # define num_access
        self.num_access = 0

        # define latency
        self.latency = param.xbar_outMem_lat

        # size equals the xbar_size, each entry being to
        self.xbar_size = xbar_size
        self.memfile = ['0' * cfg.xbdata_width] * self.xbar_size
        self.wr_pointer = 0

    def getLatency (self):
        return self.latency

    def read (self, addr):
        self.num_access += 1
        assert (type(addr) == int), 'addr type should be int'
        assert (-1 < addr < self.xbar_size), 'addr exceeds the memory bounds'
        return self.memfile[addr]

    # reads entire xbar_out_mem in parallel
    def read_p (self):
        # Fix self.num_access_read ??
        #self.num_access_read += 1
        out_list = []
        for i in range(self.xbar_size):
            value = self.memfile[i]
            out_list.append(value)
        return out_list

    def write_n (self, addr, data):
        self.num_access += 1
        assert (type(addr) == int), 'addr type should be int'
        assert (-1 < addr < self.xbar_size), 'addr exceeds the memory bounds'
        assert ((type(data) ==  str) and (len(data) == cfg.xbdata_width)), 'data should be a string with xbdata_width bits'
        self.memfile[addr] = data

    def write (self, data):
        self.num_access += 1
        assert ((type(data) ==  str) and (len(data) == cfg.xbdata_width)), 'data should be a string with xbdata_width bits'
        self.memfile[self.wr_pointer] = data
        self.wr_pointer = self.wr_pointer + 1

    def restart (self):
        # self.num_access += 1
        self.wr_pointer = 0

    def reset (self):
        # self.num_access += 1
        self.memfile = ['0' * cfg.xbdata_width] * self.xbar_size
        self.wr_pointer = 0


# Instruction memory stores dict unlike memory (string)
class instrn_memory (memory):

    def __init__ (self, size, addr_offset = 0):
        # define num_access
        self.num_access = 0

        # define latency
        self.latency = param.instrnMem_lat

        # memfile will store half-word (16 bits digital data) length strings
        self.size = size
        self.memfile = [''] * size
        self.addr_start = addr_offset
        self.addr_end = self.addr_start + self.size -1

    # To initilzie the memory with instructions
    def load (self, dict_list):
        if (len(dict_list) > self.size):
            print(('instruction memory size requirement', len(dict_list), self.size))
        assert (len(dict_list) <= self.size), 'instructions exceed the instruction memory size'
        for i in range(len(dict_list)):
            self.memfile[i] = dict_list[i]

    def read (self, addr):
        self.num_access += 1
        assert (type(addr) == int), 'addr type should be int'
        assert (-1 < addr < self.size), 'addr exceeds the memory bounds'
        if (type(self.memfile[addr]) == dict):
            return self.memfile[addr].copy()
        else:
            return self.memfile[addr]

    def write (self, addr, data):
        self.num_access += 1
        assert (type(addr) == int), 'addr type should be int'
        assert (-1 < addr < self.size), 'addr exceeds the memory bounds'
        assert (type(data) == dict), 'instrn should of type dictionary'
        self.memfile[addr] = data
        return 1


# Memory interface to interact with an external memory
class mem_interface (object):
    def __init__ (self):
        # define latency
        self.latency = param.memInterface_lat

        # in/out ports
        self.wait = 0  # wait signal from (EDRAM) controller to ima
        self.ren = 0  # ren = 1, for LD
        self.wen = 0  # wen = 1, for ST
        self.wr_width = 0
        self.rd_width = 0
        self.addr = 0 # add sent by ima to mem controller
        self.ramload = 0 # data (for LD) sent by edram to ima
        self.ramstore = 0 # data (for ST) sent by ima to mem controller

        ## For DEBUG of IMA only - define a memory element and preload some values
        #self.edram = memory (cfg.dataMem_size, 0)
        #for i in range (len(self.edram.memfile)/2):
        #    val = int2bin (i, cfg.data_width)
        #    self.edram.memfile[i] = val

    def getLatency (self):
        return self.latency

    def wrRequest (self, addr, ramstore, wr_width):
        assert (type(ramstore[1][0]) == str), 'data type expected string'
        self.wen = 1
        self.ren = 0
        self.wr_width = wr_width
        self.addr = addr
        self.ramstore = ramstore
        self.wait = 1

        ## For DEBUG of IMA only
        #self.edram.memfile[addr] = ramstore

    def rdRequest (self, addr, rd_width):
        self.ren = 1
        self.wen = 0
        self.rd_width = rd_width
        self.addr = addr
        self.wait = 1

        ## For DEBUG of IMA only
        #self.ramload = self.edram.memfile[addr]

'''
class MemRequestType(Enum):
    READ = 1
    WRITE = 2
    ALLOC = 3
    FREE = 4

class MemoryRequest:
    def __init__(self, 
                 req_type: MemRequestType,
                 core_id: int,
                 block_addr: Optional[int] = None,
                 data: Optional[bytes] = None,
                 size: int = 0):
        self.req_type = req_type
        self.core_id = core_id
        self.block_addr = block_addr
        self.data = data
        self.size = size
        self.timestamp = 0
'''