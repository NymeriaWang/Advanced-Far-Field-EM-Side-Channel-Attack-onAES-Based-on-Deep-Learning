import collections
import enum
import time
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import random
import AES as ENCRYPT
import produce_round_key


AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])


def encoder(pt_save, ct_save, key_save, roundkey10_save, keylist_save, trace_save, nor_trace_maxmin_save, plaintext_path, key_path, trace_path, real_index, select):
    number_from_select = 10
    averaging = 1
   
    # read trace data-----------------------------------------------
    Trace = []
    nor_trace_maxmin = []
    nor_trace_meanstd = []

    
    #print(list(missing_index))
    
    #print(real_index[3962])
    
    for i in tqdm(select[:], ncols=60):
    
        for j in range(number_from_select):
            
            path = trace_path + str(i) + '.npy'
            all_trace = np.load(path)
            
            #trace_select = j
            trace_select = [k for k in range(j,j+averaging)]
            
            one_trace = all_trace[trace_select].mean(axis=0)
            #print(one_trace.shape)
            Trace.append(one_trace)
    
        # for max-min normalization
            MAX = np.max(one_trace)
            MIN = np.min(one_trace)
        
            nor_one_trace_maxmin = (one_trace-MIN)/(MAX-MIN)
            nor_trace_maxmin.append(nor_one_trace_maxmin)

        
    Trace = np.array(Trace)#[select]
    nor_trace_maxmin = np.array(nor_trace_maxmin)#[select]
    #nor_trace_meanstd = np.array(nor_trace_meanstd)[select]
    
    print('trace shape:', Trace.shape)
    
    #np.save(trace_save, Trace)  # for chipwhisperer and machine learning
    np.save(nor_trace_maxmin_save, nor_trace_maxmin)  # for machine learning
    #np.save(nor_trace_meanstd_save, nor_trace_meanstd)  # for machine learning
    
    # read plaintext data-----------------------------------------------
    
    
    # read plaintext data-----------------------------------------------
    
    plaintext = []
    
    with open(plaintext_path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  # delete /n
            plaintext.append(line)
    
    for i in range(len(plaintext)):
    
        temp = []
    
        for j in range(16):
            byte = plaintext[i][2 * j: 2 * j + 2]
            dec = int(byte, 16)
            temp.append(dec)
    
        plaintext[i] = temp
    
    plaintext = np.array(plaintext)
    
    #plaintext = plaintext[real_index]
    plaintext = plaintext[select]
    plaintext = plaintext.repeat(number_from_select,axis=0)
    
    
    print('pt shape:', plaintext.shape)
    
    np.save(pt_save, plaintext)  # for chipwhisperer
    
    
    
    # read key data-----------------------------------------------
    
    
    with open(key_path, 'r') as f:
        data = f.read()
    
    key = []
    for i in range(16):
        byte = data[2 * i: 2 * i + 2]
        dec = int(byte, 16)
        key.append(dec)
    
    key = np.array(key)
    
    hex_key = []
    for x in key:
        hex_key.append(hex(x))
    
    print('hex_key:',hex_key)
    
    key_list = []
    for i in range(len(select)):
        key_list.append(key)
    
    key_list = np.array(key_list)
    key_list = key_list.repeat(number_from_select,axis=0)
    print('keylist length:', len(key_list))
    np.save(key_save, key)  # for chipwhisperer
    np.save(keylist_save, key_list)  # for chipwhisperer
    
    
    # get ciphertext --------------------------------------------------------
    
    ciphertexts = []
    
    
    str_key = ''
    for i in range(len(key)):
        str_key = str_key + format(key[i], '#04x').split('x')[-1]
    
    print('str_key:' , str_key)
    
    for pt in tqdm(plaintext,ncols=60):
    
        str_pt = ''
    
        for i in range(len(pt)):
            str_pt = str_pt + format(pt[i], '#04x').split('x')[-1]
    
        KEY = str_key
        DATA = str_pt
    
        aes = ENCRYPT.AES(mode='ecb', input_type='hex')
        ct = aes.encryption(DATA, KEY)
    
        ciphertexts.append(ct)
    
    for i in range(len(ciphertexts)):
    
        temp = []
    
        for j in range(16):
            byte = ciphertexts[i][2 * j: 2 * j + 2]
            dec = int(byte, 16)
            temp.append(dec)
    
        ciphertexts[i] = temp
    
    ciphertexts = np.array(ciphertexts)
    #ciphertexts = ciphertexts.repeat(number_from_select,axis=0)
    np.save(ct_save,ciphertexts)
    
    
    # get round key and get label of sbox out in the LAST round----------------------------------------------------
    roundkey = []
    for i in range(len(key_list)):
        one_roundkey = produce_round_key.keyScheduleRounds(key_list[i], 0, 10)
        roundkey.append(one_roundkey)
    roundkey = np.array(roundkey)
    #roundkey= roundkey.repeat(number_from_select,axis=0)
    print('10th roundkey shape:', roundkey.shape)
    
    np.save(roundkey10_save,roundkey)


def decoder(pt_save, ct_save, key_save, roundkey10_save, keylist_save, trace_save, nor_trace_maxmin_save, plaintext_path, key_path, trace_path, real_index, select):
    number_from_select = 10
    averaging = 100    
       
    
    # read trace data-----------------------------------------------
    Trace = []
    nor_trace_maxmin = []
    nor_trace_meanstd = []
    

    
    #print(list(missing_index))
    
    #print(real_index[3962])
    
    for i in tqdm(select[:], ncols=60):
        path = trace_path + str(i) + '.npy'
        all_trace = np.load(path)
    
        trace_select = random.sample(range(0, len(all_trace)), averaging)
    
        one_trace = all_trace[trace_select].mean(axis=0)
    
        Trace.append(one_trace)
    
        # for max-min normalization
        MAX = np.max(one_trace)
        MIN = np.min(one_trace)
        
        nor_one_trace_maxmin = (one_trace-MIN)/(MAX-MIN)
        nor_trace_maxmin.append(nor_one_trace_maxmin)    
    

        
    Trace = np.array(Trace)#[select]
    Trace = Trace.repeat(number_from_select,axis=0)
    nor_trace_maxmin = np.array(nor_trace_maxmin)#[select]
    #nor_trace_meanstd = np.array(nor_trace_meanstd)[select]
    nor_trace_maxmin = nor_trace_maxmin.repeat(number_from_select,axis=0)
    print('trace shape:', Trace.shape)
    
    np.save(trace_save, Trace)  # for chipwhisperer and machine learning
    np.save(nor_trace_maxmin_save, nor_trace_maxmin)  # for machine learning
    #np.save(nor_trace_meanstd_save, nor_trace_meanstd)  # for machine learning
    
    # read plaintext data-----------------------------------------------
    
    
    # read plaintext data-----------------------------------------------
    
    plaintext = []
    
    with open(plaintext_path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  # delete /n
            plaintext.append(line)
    
    for i in range(len(plaintext)):
    
        temp = []
    
        for j in range(16):
            byte = plaintext[i][2 * j: 2 * j + 2]
            dec = int(byte, 16)
            temp.append(dec)
    
        plaintext[i] = temp
    
    plaintext = np.array(plaintext)
    
    #plaintext = plaintext[real_index]
    plaintext = plaintext[select]
    plaintext = plaintext.repeat(number_from_select,axis=0)
    
    
    print('pt shape:', plaintext.shape)
    
    np.save(pt_save, plaintext)  # for chipwhisperer
    
    
    
    # read key data-----------------------------------------------
    
    
    with open(key_path, 'r') as f:
        data = f.read()
    
    key = []
    for i in range(16):
        byte = data[2 * i: 2 * i + 2]
        dec = int(byte, 16)
        key.append(dec)
    
    key = np.array(key)
    
    hex_key = []
    for x in key:
        hex_key.append(hex(x))
    
    print('hex_key:',hex_key)
    
    key_list = []
    for i in range(len(select)):
        key_list.append(key)
    
    key_list = np.array(key_list)
    key_list = key_list.repeat(number_from_select,axis=0)
    print('keylist length:', len(key_list))
    np.save(key_save, key)  # for chipwhisperer
    np.save(keylist_save, key_list)  # for chipwhisperer
    
    
    # get ciphertext --------------------------------------------------------
    
    ciphertexts = []
    
    
    str_key = ''
    for i in range(len(key)):
        str_key = str_key + format(key[i], '#04x').split('x')[-1]
    
    print('str_key:' , str_key)
    
    for pt in tqdm(plaintext,ncols=60):
    
        str_pt = ''
    
        for i in range(len(pt)):
            str_pt = str_pt + format(pt[i], '#04x').split('x')[-1]
    
        KEY = str_key
        DATA = str_pt
    
        aes = ENCRYPT.AES(mode='ecb', input_type='hex')
        ct = aes.encryption(DATA, KEY)
    
        ciphertexts.append(ct)
    
    for i in range(len(ciphertexts)):
    
        temp = []
    
        for j in range(16):
            byte = ciphertexts[i][2 * j: 2 * j + 2]
            dec = int(byte, 16)
            temp.append(dec)
    
        ciphertexts[i] = temp
    
    ciphertexts = np.array(ciphertexts)
    #ciphertexts = ciphertexts.repeat(number_from_select,axis=0)
    np.save(ct_save,ciphertexts)
    
    
    # get round key and get label of sbox out in the LAST round----------------------------------------------------
    roundkey = []
    for i in range(len(key_list)):
        one_roundkey = produce_round_key.keyScheduleRounds(key_list[i], 0, 10)
        roundkey.append(one_roundkey)
    roundkey = np.array(roundkey)
    #roundkey= roundkey.repeat(number_from_select,axis=0)
    print('10th roundkey shape:', roundkey.shape)
    
    np.save(roundkey10_save,roundkey)


def averaging_trace(pt_save, ct_save, key_save, roundkey10_save, keylist_save, trace_save, nor_trace_maxmin_save, plaintext_path, key_path, trace_path, real_index, select):
    number_from_select = 10
    averaging = 1
       
    
    # read trace data-----------------------------------------------
    Trace = []
    nor_trace_maxmin = []
    nor_trace_meanstd = []
    

    
    #print(list(missing_index))
    
    #print(real_index[3962])
    
    for i in tqdm(select[:], ncols=60):
        path = trace_path + str(i) + '.npy'
        all_trace = np.load(path)
    
        trace_select = random.sample(range(0, len(all_trace)), averaging)
    
        one_trace = all_trace[trace_select].mean(axis=0)
    
        Trace.append(one_trace)
    
        # for max-min normalization
        MAX = np.max(one_trace)
        MIN = np.min(one_trace)
        
        nor_one_trace_maxmin = (one_trace-MIN)/(MAX-MIN)
        nor_trace_maxmin.append(nor_one_trace_maxmin)    
    

        
    Trace = np.array(Trace)#[select]
    nor_trace_maxmin = np.array(nor_trace_maxmin)#[select]
    #nor_trace_meanstd = np.array(nor_trace_meanstd)[select]
    print('trace shape:', Trace.shape)
    
    #np.save(trace_save, Trace)  # for chipwhisperer and machine learning
    np.save(nor_trace_maxmin_save, nor_trace_maxmin)  # for machine learning
    #np.save(nor_trace_meanstd_save, nor_trace_meanstd)  # for machine learning
    
    # read plaintext data-----------------------------------------------
    
    
    # read plaintext data-----------------------------------------------
    
    plaintext = []
    
    with open(plaintext_path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  # delete /n
            plaintext.append(line)
    
    for i in range(len(plaintext)):
    
        temp = []
    
        for j in range(16):
            byte = plaintext[i][2 * j: 2 * j + 2]
            dec = int(byte, 16)
            temp.append(dec)
    
        plaintext[i] = temp
    
    plaintext = np.array(plaintext)
    
    #plaintext = plaintext[real_index]
    plaintext = plaintext[select]   
    
    print('pt shape:', plaintext.shape)
    
    np.save(pt_save, plaintext)  # for chipwhisperer
    
    
    
    # read key data-----------------------------------------------
    
    
    with open(key_path, 'r') as f:
        data = f.read()
    
    key = []
    for i in range(16):
        byte = data[2 * i: 2 * i + 2]
        dec = int(byte, 16)
        key.append(dec)
    
    key = np.array(key)
    
    hex_key = []
    for x in key:
        hex_key.append(hex(x))
    
    print('hex_key:',hex_key)
    
    key_list = []
    for i in range(len(select)):
        key_list.append(key)
    
    key_list = np.array(key_list)
    print('keylist length:', len(key_list))
    np.save(key_save, key)  # for chipwhisperer
    np.save(keylist_save, key_list)  # for chipwhisperer
    
    
    # get ciphertext --------------------------------------------------------
    
    ciphertexts = []
    
    
    str_key = ''
    for i in range(len(key)):
        str_key = str_key + format(key[i], '#04x').split('x')[-1]
    
    print('str_key:' , str_key)
    
    for pt in tqdm(plaintext,ncols=60):
    
        str_pt = ''
    
        for i in range(len(pt)):
            str_pt = str_pt + format(pt[i], '#04x').split('x')[-1]
    
        KEY = str_key
        DATA = str_pt
    
        aes = ENCRYPT.AES(mode='ecb', input_type='hex')
        ct = aes.encryption(DATA, KEY)
    
        ciphertexts.append(ct)
    
    for i in range(len(ciphertexts)):
    
        temp = []
    
        for j in range(16):
            byte = ciphertexts[i][2 * j: 2 * j + 2]
            dec = int(byte, 16)
            temp.append(dec)
    
        ciphertexts[i] = temp
    
    ciphertexts = np.array(ciphertexts)
    np.save(ct_save,ciphertexts)
    
    
    # get round key and get label of sbox out in the LAST round----------------------------------------------------
    roundkey = []
    for i in range(len(key_list)):
        one_roundkey = produce_round_key.keyScheduleRounds(key_list[i], 0, 10)
        roundkey.append(one_roundkey)
    roundkey = np.array(roundkey)
    print('10th roundkey shape:', roundkey.shape)
    
    np.save(roundkey10_save,roundkey)
  


save_folder = 'd6_averaged_trace/'

pt_save = save_folder + 'pt.npy'
ct_save = save_folder + 'ct.npy'
key_save = save_folder + 'key.npy'
roundkey10_save = save_folder + '10th_roundkey.npy'
keylist_save = save_folder + 'keylist.npy'
trace_save = save_folder + 'traces.npy'
nor_trace_maxmin_save = save_folder + 'nor_traces_maxmin.npy'
#nor_trace_meanstd_save = save_folder + 'nor_traces_meanstd.npy'
#label_save = save_folder + 'label_lastround_Sout_0.npy'


#===========================================================
load_folder = 'original_data'
mis_index_path = load_folder + '/mis_index.npy'
plaintext_path = load_folder + '/pt_.txt'
key_path = load_folder + '/key_.txt'
trace_path = load_folder + '/all__'

total_nunmber = 5100
select_number = 5000


total_index = [i for i in range(total_nunmber)]
missing_index = np.load(mis_index_path)
real_index = [x for x in total_index if x not in missing_index]
real_index =np.array(real_index)
#select_index = real_index[[i for i in range(select_number)]]
select = real_index[[i for i in range(select_number)]]
#select = np.array(select)
#random.sample(range(len(real_index)), select_number)




encode_discriminant=2  #0: encode, 1: decode, 2: averaging




if encode_discriminant == 0:
   
    encoder(pt_save, ct_save, key_save, roundkey10_save, keylist_save, trace_save, nor_trace_maxmin_save, plaintext_path, key_path, trace_path, real_index, select)
 
    
elif encode_discriminant == 1:
    
    decoder(pt_save, ct_save, key_save, roundkey10_save, keylist_save, trace_save, nor_trace_maxmin_save, plaintext_path, key_path, trace_path, real_index, select)
     
    
elif encode_discriminant == 2:
    averaging_trace(pt_save, ct_save, key_save, roundkey10_save, keylist_save, trace_save, nor_trace_maxmin_save, plaintext_path, key_path, trace_path, real_index, select)

else:
    print("Error: provided encode discriminant does not exist!")
    sys.exit(-1)












