import enum

## data config (fixed point allowed only)
class num_config(enum.Enum):

    # storage_config illustrates how data is storaged
    # e.g. ['2', '2', '2', '2', '2', '1', '1', 's', 's', 's', 's']: a 16 bit parameter, 10 lower bits storaged in 5 2-bit ReRAM MLCs, 2 middle bits storaged in 2 ReRAM SLCs, 4 higher bits storaged in 4 SRAMs
    storage_config = ['2', '2', '2', '2', '2', '2', '2', '2'] # storaged in 8 2-bit ReRAM MLCs
     
    # storage_bit marks the start bit each xbar is storaging. e.g. [0, 2, 4, 6, 8, 10, 11, 12, 13, 14, 15] will be generated for the example above.
    # this means the first xbar storages the 0th and 1st bit, second storages 2nd and 3rd bit etc.
    storage_bit = []
    
    num_bits = 0 # total bits number in the operand
    ReRAM_xbar_num = 0 # number of ReRAM xbars
    SRAM_xbar_num = 0 # number of SRAM xbars
    for i in storage_config:
        storage_bit.append(num_bits)
        if i == 's':
            num_bits += 1
            SRAM_xbar_num += 1
        else:
            num_bits += int(i)
            ReRAM_xbar_num += 1

    int_bits = 12
    assert int_bits <= num_bits, 'storage config invalid: int_bits is more than total storaged bits'
    frac_bits = num_bits - int_bits