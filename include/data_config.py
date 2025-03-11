## data config (fixed point allowed only)
class num_config:
    def __init__(self, storage_config = ['2', '2', '2', '2', '2', '2', '2', '2'], int_bits = 12):
        # storage_config illustrates how data is stored
        # e.g. ['2', '2', '2', '2', '2', '1', '1', 's', 's', 's', 's']: a 16 bit parameter, 10 lower bits stored in 5 2-bit ReRAM MLCs, 2 middle bits stored in 2 ReRAM SLCs, 4 higher bits stored in 4 SRAMs
        self.storage_config = storage_config # stored in 8 2-bit ReRAM MLCs
     
        # storage_bit marks the start bit each xbar is storaging. e.g. [0, 2, 4, 6, 8, 10, 11, 12, 13, 14, 15] will be generated for the example above.
        # this means the first xbar storages the 0th and 1st bit, second storages 2nd and 3rd bit etc.

        self.stored_bit = []
        self.bits_per_cell = []
    
        self.num_bits = 0 # total bits number in the operand
        self.ReRAM_xbar_num = 0 # number of ReRAM xbars
        self.SRAM_xbar_num = 0 # number of SRAM xbars
        for i in self.storage_config:
            self.stored_bit.append(self.num_bits)
            if i == 's':
                self.num_bits += 1
                self.SRAM_xbar_num += 1
                self.bits_per_cell.append(1)
            else:
                self.num_bits += int(i)
                self.ReRAM_xbar_num += 1
                self.bits_per_cell.append(int(i))

        self.int_bits = int_bits
        assert self.int_bits <= self.num_bits, 'storage config invalid: int_bits is more than total stored bits'
        self.frac_bits = self.num_bits - self.int_bits



# Change here to use different data config.
# the first param indicates how data is stored.
# the second param indicates how many int_bits are included.
# e.g. ['2', '2', '2', '2', '2', '1', '1', 's', 's', 's', 's'], 12: 
# a 16 bit parameter, 10 lower bits stored in 5 2-bit ReRAM MLCs, 2 middle bits stored in 2 ReRAM SLCs, 4 higher bits stored in 4 SRAMs
# 12 int bit, 4 frac bit
datacfg_list = [num_config(['2', '2', '2', '2', '2', '2', '2', '2'], 8),
                num_config(['1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1'], 8),
                num_config(['3', '3', '3', '3', '2', '2'], 8),
                num_config(['4', '4', '4', '4'], 8),
                num_config(['1', '1', '1', '1', '2', '2', '2', '2', '2', '2'], 8),
                num_config(['4', '2', '2', '2', '2', '1', '1', '1', '1'], 8)]

datacfg = datacfg_list[0]

max_val = 2.0 ** (datacfg.int_bits - 1) - 1.0 / (2.0 ** datacfg.frac_bits)
min_val = -2.0 ** (datacfg.int_bits - 1)
