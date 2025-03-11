import sys
import getopt
import os
import argparse
import subprocess
import threading
from multiprocessing import Pool, Queue, Process, Manager
import time
import psutil
from functools import partial
from queue import Queue

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
simulator_dir = os.path.join(root_dir, "srcs/Hybrid-CiM-Simulator")
data_dir = os.path.join(root_dir, "data")
src_dir = os.path.join(simulator_dir, "src")
include_dir = os.path.join(simulator_dir, "include")
security_dir=os.path.join(root_dir, "Security")

sys.path.insert(1, security_dir)
sys.path.insert(0, include_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, simulator_dir)

from data_convert import *
from node_dump import *
from record_xbar import *
from hw_stats import *
import numpy as np

import argparse
import config as cfg
from data_config import datacfg
import constants
import ima_modules
import ima
import tile_modules
import tile
import node_modules
import node
import ima_metrics
import tile_metrics
import node_metrics
import dnn_wt_p

compiler_path = '/HybridCiM/data/testasm/'
trace_path = '/HybridCiM/data/traces/'

def modify_datacfg(config_id):
    with open('../include/data_config.py', 'r') as file:
        lines = file.readlines()

    # Modify the line containing the variable
    for i, line in enumerate(lines):
        if line.startswith('datacfg = '):
            lines[i] = "datacfg = datacfg_list[{}]\n".format(config_id)

    # Write the changes back to the file
    with open('../include/data_config.py', 'w') as file:
        file.writelines(lines)
    
def modify_tile_num(tile_num):
    with open('../include/config.py', 'r') as file:
        lines = file.readlines()

    # Modify the line containing the variable
    for i, line in enumerate(lines):
        if line.startswith('num_tile_compute = '):
            lines[i] = "num_tile_compute = {}\n".format(tile_num)

    # Write the changes back to the file
    with open('../include/config.py', 'w') as file:
        file.writelines(lines)

def count_tiles(net_path):
    t_count = 0
    while os.path.isdir(net_path + "/tile" + str(t_count)):
        t_count += 1
    return t_count

def check_RAM():
    ram = psutil.virtual_memory()
    return ram.available > 256 * 1024 * 1024 * 1024  # 256GB in bytes

def preprocess(net):
    print(("Preprocessing net {}.".format(net)))
    instrndir = compiler_path + net
    
    # Run generate-py.sh
    try:
        result = subprocess.run(
            ['/bin/bash', '../script/generate-py.sh', instrndir],
            capture_output=True,
            text=True,
            check=False
        )
    
        if result.returncode != 0:
            print("Error in generate-py.sh:")
            print(result.stderr)
            sys.exit(1)
        print(result.stdout)
    except Exception as e:
        print(f"Failed to execute script: {e}")
        sys.exit(1)
    '''
    cmd1 = ['/bin/bash', '../script/generate-py.sh', instrndir]
    p1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout1, stderr1 = p1.communicate()
    if p1.returncode != 0:
        print("Error in generate-py.sh:")
        print(stderr1)
        sys.exit(1)
    print(stdout1)
    '''
    
    # Run populate.py
    cmd2 = ['python', 'populate.py', '--path', instrndir]
    p2 = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout2, stderr2 = p2.communicate()
    if p2.returncode != 0:
        print("Error in populate.py:")
        print(stderr2)
        sys.exit(1)
    print(stdout2)
    print(("Completed preprocessing net {}.".format(net)))

def run_dpe(net, inp, instance_id):
    print(("Starting DPE instance {}.".format(instance_id)))

    instrndir = compiler_path + net
    tracedir = trace_path + net

    assert (os.path.exists(instrndir+'/'+'tile0')), 'Input Error: Provide input before running the DPE'
    assert (os.path.exists(instrndir) ==1), 'Instructions for net missing: generate intuctions (in folder hierarchy) hierarchy'

    if not os.path.exists(tracedir):
        os.makedirs(tracedir)

    for i in range(cfg.num_tile):
        temp_tiledir = tracedir + '/tile' + str(i)
        if not os.path.exists(temp_tiledir):
            os.makedirs(temp_tiledir)

    instrnpath = instrndir + '/'
    tracepath = tracedir + '/'

    node_dut = node.node()
    node_dut.node_init(instrnpath, tracepath)

    inp_tileId = 0
    out_tileId = 1

    for i in range(len(inp['data'])):
        data = float2fixed(inp['data'][i], datacfg.int_bits, datacfg.frac_bits)
        node_dut.tile_list[inp_tileId].edram_controller.mem.memfile[i] = data
        node_dut.tile_list[inp_tileId].edram_controller.counter[i] = int(inp['counter'][i])
        node_dut.tile_list[inp_tileId].edram_controller.valid[i] = int(inp['valid'][i])

    dnn_wt_p.dnn_wt().prog_dnn_wt(instrnpath, node_dut)

    cycle = 0
    while (not node_dut.node_halt and cycle < cfg.cycles_max):
        node_dut.node_run(cycle)
        cycle = cycle + 1
        '''
        if (instance_id % cfg.thread_num == 0):
            print ('Cycle: ', cycle, 'Tile halt list', node_dut.tile_halt_list)
        '''
    
    memfile = node_dut.tile_list[out_tileId].edram_controller.mem.memfile
    output = []

    for addr in range(len(memfile)):
        # to print in float format
        if (memfile[addr] != ''):
            temp_val = fixed2float (memfile[addr], datacfg.int_bits, datacfg.frac_bits)
            output.append(temp_val)
    
    if instance_id == 0:
        hwtrace_file = tracepath + 'harwdare_stats.txt'
        fid = open(hwtrace_file, 'w')
        metric_dict = get_hw_stats(fid, node_dut, cycle)
        fid.close()

    print(("Completed DPE instance {}.".format(instance_id)))

    result = {'instance_id': instance_id, 'output': output}
    return result

def run_dpe_wrapper(args):
    net, input_data, instance_id = args
    if not check_RAM():
        print("Warning: Not enough RAM available to run DPE instance, waiting...")
        while not check_RAM():
            time.sleep(100)
    return run_dpe(net, input_data, instance_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--net", help="The net name as it is in test/testasm.", default='fc_layer')
    parser.add_argument(
            "-t", "--tile", help ="The number of tiles as generated by compiler.", default=-1)
    parser.add_argument(
            '-c',"--cryptography", help="Run an encrypted model")
    parser.add_argument(
            '-a',"--authenticated", help="Run only authenticated models") 
    parser.add_argument(
            '-d',"--datacfg", help="The data configuration to use. Shoule be from 0 to 6. See in include/data_config.py", default=0)
    parser.add_argument(
            '-b',"--dataset", help="The dataset to process", default='none')
    args = parser.parse_args()
    
    modify_datacfg(int(args.datacfg))
    preprocess(args.net)

    model_path = os.path.join(compiler_path, args.net)

    total_tiles = count_tiles(model_path) - 2
    print(("Total tiles: {}".format(total_tiles)))

    if(args.tile != -1):
        total_tiles = int(args.tile)

    modify_tile_num(total_tiles)

    if args.dataset == 'none':
        cfg.debug = True
        os.system('python dpe.py -n ' + args.net)
    else:
        cfg.debug = False
        manager = Manager()
        results = manager.list()
        inp = np.load('/HybridCiM/data/dataset/' + args.dataset + '_input.npy', allow_pickle=True, encoding='latin1')
        labels = np.load('/HybridCiM/data/dataset/' + args.dataset + '_labels.npy', allow_pickle=True, encoding='latin1')
        #total_inputs = len(inp)
        total_inputs = 8

        for i in range(total_inputs):
            inp[i]['data'] = quantize_to_fixed(inp[i]['data'])

        args_list = []
        for i in range(total_inputs):
            if i < len(inp):
                args_list.append((args.net, inp[i], i))
        

        pool = Pool(processes = min(cfg.thread_num, total_inputs))
        process_results = pool.map(run_dpe_wrapper, args_list)
        pool.close()
        pool.join()

        results.extend(process_results)
        results = list(results)
        print("All DPE instances completed.")
        #print(results)
        

        # sort results by instance_id
        results.sort(key=lambda x: x['instance_id'])

        correct = 0

        for i in range(total_inputs):
            if np.argmax(results[i]['output']) == labels[results[i]['instance_id']]:
                correct += 1
        
        accuracy = correct / total_inputs
        print(("Accuracy: {}".format(accuracy)))


        output_file = trace_path + args.net + '/sim_output.txt'
        fid = open(output_file, 'w')
        fid.write("Accuracy: {}\n\n".format(accuracy))
        for i in range(total_inputs):
            fid.write("Input {}\n".format(results[i]['instance_id']))
            fid.write("Output: {}\n".format(results[i]['output']))
            fid.write("Label: {}\n\n".format(labels[results[i]['instance_id']]))
        fid.close()
        
            