import polar_coding_exceptions as pcexc
import polar_coding_functions as pcfun
from rate_profile import rateprofile
import copy
import numpy as np
import csv
import math



class PolarCode:
    """Represent constructing polar codes,
    encoding and decoding messages with polar codes"""

    def __init__(self, N, K, construct, dSNR, rprofile):
        if K >= N:
            raise pcexc.PCLengthError
        elif pcfun.np.log2(N) != int(pcfun.np.log2(N)):
            raise pcexc.PCLengthDivTwoError
        else:
            self.codeword_length = N
            self.log2_N = int(math.log2(N))
            self.nonfrozen_bits = K
            self.information_size = K
            self.designSNR = dSNR
            self.n = int(np.log2(self.codeword_length))
            #self.bitrev_indices = np.array([pcfun.bitreversed(j, self.n) for j in range(N)])
            self.bitrev_indices = [pcfun.bitreversed(j, self.n) for j in range(N)]
            #self.polarcode_mask = pcfun.rm_build_mask(N, K, dSNR) if construct=="rm" else pcfun.RAN87_build_mask(N, K, dSNR) if  construct=="ran87" else pcfun.build_mask(N, K, dSNR)
            self.rprofile = rprofile
            self.polarcode_mask = self.rprofile.build_mask(construct) #in bit-reversal order
            self.rate_profile = self.polarcode_mask[self.bitrev_indices] #in decoding order
            self.frozen_bits = (self.polarcode_mask + 1) % 2  #in bitrevesal order
            self.critical_set_flag = self.rprofile.critical_set_flag((self.polarcode_mask + 1) % 2)
            self.critical_set = pcfun.generate_critical_set((self.polarcode_mask + 1) % 2)
            self.LLRs = np.zeros(2 * self.codeword_length - 1, dtype=float)
            self.BITS = np.zeros((2, self.codeword_length - 1), dtype=int)
            self.stem_LLRs = np.zeros(2 * self.codeword_length - 1, dtype=float)
            self.stem_BITS = np.zeros((2, self.codeword_length - 1), dtype=int)
            #self.list_size = L
            #self.curr_list_size = 1
            self.exp_step = 0
            self.corr_path_exist = 1

            self.dLLR_thresh = 3
            self.last_seen = []
            self.shift_locs = []
            self.PM_last = 0
            self.Loc_last = 0
            self.repeat = False
            self.window_shifted = False
            self.shift_set = []
            self.shft_idx = 0
            self.shft_pmr = []
            #self.model = load_model('model_y_10Kerr_n9_R05_L2_6in_batch418_seq2_binary_epoch30.h5')
            self.cs_seg_cnt = []
            self.seg_tot = 1
            self.flip_cnt = 0
            self.flips_const = 5
            self.bit_idx_B_updating = 0
           #list([iterbale]) is the list constructor
            self.modu = 'BPSK'
            
            self.ml_exploring_mu_min_idx = 0
            self.ml_last_mu_max = 0

            self.A = pcfun.A(self.polarcode_mask, N, K)
            self.pe = np.zeros(N, dtype=float) #pcfun.pe_dega(self.codeword_length,
                                                   #self.information_size,
                                                   #self.designSNR)
            self.sigma = 0
            self.snrb_snr = 'SNRb'
            self.Delta = 0
            self.T = 0
            self.iter_clocks = 0
            self.total_clocks = 0
            self.max_clocks = 5000
            self.total_steps = 0
            self.total_additions = 0
            self.total_comparisons = 0
            self.iter = 0
            self.err_init = 0
            self.prnt_proc = 0
            #Collecting statistics:
            self.bit_err_cnt = np.zeros(N, dtype=int)
            self.tot_err_freq = np.zeros(10, dtype=int)
            

    def __repr__(self):
        return repr((self.codeword_length, self.information_size, self.designSNR))
#__str__ (read as "dunder (double-underscore) string") and __repr__ (read as "dunder-repper" (for "representation")) are both special methods that return strings based on the state of the object.

    def mul_matrix(self, precoded):
        """multiplies message of length N with generator matrix G"""
        """Multiplication is based on factor graph"""
        N = self.codeword_length
        polarcoded = precoded
        for i in range(self.n):
            if i == 0:
                polarcoded[0:N:2] = (polarcoded[0:N:2] + polarcoded[1:N:2]) % 2
            elif i == (self.n - 1):
                polarcoded[0:int(N/2)] = (polarcoded[0:int(N/2)] + polarcoded[int(N/2):N]) % 2
            else:
                enc_step = int(pcfun.np.power(2, i))
                for j in range(enc_step):
                    polarcoded[j:N:(2 * enc_step)] = (polarcoded[j:N:(2 * enc_step)]
                                                    + polarcoded[j + pcfun.np.power(2, i):N:(2 * enc_step)]) % 2
        return polarcoded
    # --------------- ENCODING -----------------------

    def profiling(self, info):
        """Apply polar code mask to information message and return profiled message"""
        profiled = pcfun.np.zeros(self.codeword_length, dtype=int) #array
        profiled[self.polarcode_mask == 1] = info
        self.trdata = copy.deepcopy(profiled)
        return profiled

    def precode(self, info):
        """Apply polar code mask to information message and return precoded message"""
        precoded = pcfun.np.zeros(self.codeword_length, dtype=int) #array
        precoded[self.polarcode_mask == 1] = info
        self.trdata = copy.deepcopy(precoded)
        return precoded
    

    def encode(self, info, issystematic: bool):
        """Encoding function"""
        # Non-systematic encoding
        encoded = self.precode(info)
        #encoded = self.precode_seg(info)
        if self.prnt_proc==1:
            print("[ ", end='')
            for i in range(self.codeword_length):
                print(encoded[pcfun.bitreversed(i,self.n)], end='')
                if (i+1)%4==0:
                    print(" ", end='')
            print("]")
        #print(encoded)
        if not issystematic:
            polarcoded = self.mul_matrix(encoded)
        # Systematic encoding based on non-systematic encoding
        else:
            polarcoded = self.mul_matrix(encoded)
            polarcoded *= self.polarcode_mask
            polarcoded = self.mul_matrix(polarcoded)
            # ns_encoded = self.mul_matrix(self.precode(info))
            # s_encoded = [self.polarcode_mask[i] * ns_encoded[i] for i in range(self.codeword_length)]
            # return self.mul_matrix(s_encoded)
        return polarcoded



    def pac_encode(self, info, conv_gen, mem):
        """Encoding function"""
        # Non-systematic encoding
        if self.prnt_proc==1:
            print("A=[ ", end='')
            for i in range(self.codeword_length):
                print(self.polarcode_mask[pcfun.bitreversed(i,self.n)], end='')
                if (i+1)%4==0:
                    print(" ", end='')
            print("]")
        V = self.precode(info)
        if self.prnt_proc==1:
            print("V=[ ", end='')
            for i in range(self.codeword_length):
                print(V[pcfun.bitreversed(i,self.n)], end='')
                if (i+1)%4==0:
                    print(" ", end='')
            print("]")
        U = pcfun.conv_encode(V, conv_gen, mem)
        if self.prnt_proc==1:
            print("U=[ ", end='')
            for i in range(self.codeword_length):
                print(U[pcfun.bitreversed(i,self.n)], end='')
                if (i+1)%4==0:
                    print(" ", end='')
            print("]")
        X = self.mul_matrix(U)
        return X


    # -------------------------- DECODING -----------------------------------

    # --- SC Decoding ---

    def updateLLR(self, position: int):
        """updates LLR values at each step of SC decoding"""
        if position == 0:
            nextlevel = self.n
        else:
            self.iter_clocks += 1
            lastlevel = (bin(position)[2:].zfill(self.n)).find('1') + 1
            #For the last stage/level when pos < N/2 and the lest in the next loop, or when pos>N/2 where just stage/level 0 is required. 
            start = int(pow(2, lastlevel - 1)) - 1
            end = int(pow(2, lastlevel) - 1) - 1
            for i in range(start, end + 1):
                self.total_additions += 1
                self.LLRs[i] = pcfun.lowerconv(self.BITS[0][i],
                                               self.LLRs[end + 2 * (i - start) + 1],
                                               self.LLRs[end + 2 * (i - start) + 2])
                #print(self.BITS[0][i], self.LLRs[end + 2 * (i - start) + 1], self.LLRs[end + 2 * (i - start) + 2])
            nextlevel = lastlevel - 1
        #When pos < N/2, the last stage is done in the previous loop and then this loop continues
        for lev in range(nextlevel, 0, -1):
            self.iter_clocks += 1
            start = int(pow(2, lev - 1)) - 1
            end = int(pow(2, lev) - 1) - 1
            for indx in range(start, end + 1):
                self.total_comparisons += 1
                self.LLRs[indx] = pcfun.upperconv(self.LLRs[end + 2 * (indx - start) + 1],
                                                  self.LLRs[end + 2 * (indx - start) + 2])
                #print(self.LLRs[end + 2 * (indx - start) + 1], self.LLRs[end + 2 * (indx - start) + 2])

    def updateBITS(self, latestbit: int, position: int):
        """updates bit values at each step of SC decoding"""
        if position == self.codeword_length - 1:
            return
        elif position < self.codeword_length // 2:
            self.BITS[0][0] = latestbit
        else:
            lastlevel = (bin(position)[2:].zfill(self.n)).find('0') + 1
            self.BITS[1][0] = latestbit
            #For auxiliary calc (all the stages befoer last stage/level), to be stored in the 2nd row
            for lev in range(1, lastlevel - 1):
                st = int(pow(2, lev - 1)) - 1
                ed = int(pow(2, lev) - 1) - 1
                for i in range(st, ed + 1):
                    self.BITS[1][ed + 2 * (i - st) + 1] = (self.BITS[0][i] + self.BITS[1][i]) % 2
                    self.BITS[1][ed + 2 * (i - st) + 2] = self.BITS[1][i]
            #For last stage/level, to be used directly in g nodes.
            lev = lastlevel - 1
            st = int(pow(2, lev - 1)) - 1
            ed = int(pow(2, lev) - 1) - 1
            for i in range(st, ed + 1):
                self.BITS[0][ed + 2 * (i - st) + 1] = (self.BITS[0][i] + self.BITS[1][i]) % 2
                self.BITS[0][ed + 2 * (i - st) + 2] = self.BITS[1][i]

    def extract(self, decoded_message):
        """Extracts bits from information positions due to polar code mask"""
        decoded_info = pcfun.np.array(list(), dtype=int)
        mask = self.polarcode_mask
        for i in range(len(self.polarcode_mask)):
            if mask[i] == 1:
                # decoded_info.append(decoded_message[i])
                decoded_info = pcfun.np.append(decoded_info, decoded_message[i])
        return decoded_info


    def sc_decode(self, soft_mess, issystematic: bool):
        """SC-decoder
        symbol_energy -  the BPSK symbol energy (linear scale);
        noise_power -  Noise power spectral density (default N0/2 = 1)"""
        # reset LLRs
        self.LLRs = [0 for i in range(2 * self.codeword_length - 1)]
        # reset BITS
        self.BITS = [[0 for i in range(self.codeword_length - 1)] for j in range(2)]
        # reset decoding results
        decoded = [0 for i in range(self.codeword_length)]
        # initial LLRs
        self.LLRs[self.codeword_length - 1:] = soft_mess
        for j in range(self.codeword_length):
            i = pcfun.bitreversed(j, self.n)
            self.updateLLR(i)
            if self.polarcode_mask[i] == 1:
                if self.LLRs[0] > 0:
                    decoded[i] = 0
                else:
                    decoded[i] = 1
            else:
                decoded[i] = self.polarcode_mask[i]
            self.updateBITS(decoded[i], i)

        if issystematic:
            self.mul_matrix(decoded)
            return self.extract(decoded)
        return self.extract(decoded)



    def updateLLR_back(self, position: int):
        #Assuming only one step has gone backward
        """updates LLR values at each step of SC decoding"""
        if position >= self.codeword_length//2: #in terms of i, they are odd, but in tems ii, they are >= N/2
            #The position of 1 from MSP tells us the starting stage of factor graph.
            #The starting stage includes g nodes.
            pos_post = pcfun.bitreversed(pcfun.bitreversed(position, self.n) + 1, self.n)
            pos_pre = pcfun.bitreversed(pcfun.bitreversed(position, self.n) - 1, self.n)
            lastlevel_post = (bin(pos_post)[2:].zfill(self.n)).find('1') + 1
            lastlevel_pre = (bin(pos_pre)[2:].zfill(self.n)).find('1') + 1
            if lastlevel_pre < lastlevel_post:
                jmp = 3
                while lastlevel_pre <= lastlevel_post:
                    pos_pre = pcfun.bitreversed(pcfun.bitreversed(position, self.n) - jmp, self.n)
                    lastlevel_pre = (bin(pos_pre)[2:].zfill(self.n)).find('1') + 1
                    jmp += 2
                   
            elif  lastlevel_pre >= lastlevel_post:
                if pos_pre == 0:
                    nextlevel = self.n
                else:
                        
                    lastlevel = (bin(pos_pre)[2:].zfill(self.n)).find('1') + 1
                    start = int(pow(2, lastlevel - 1)) - 1
                    end = int(pow(2, lastlevel) - 1) - 1
                    for i in range(start, end + 1):
                        self.LLRs[i] = pcfun.lowerconv(self.BITS[0][i],
                                                       self.LLRs[end + 2 * (i - start) + 1],
                                                       self.LLRs[end + 2 * (i - start) + 2])
                    nextlevel = lastlevel - 1
                #when bitreversal of position is >=N/2, because of lastlevel=1, the proceeding loop does not get through
                for lev in range(nextlevel, 0, -1):
                    if lev > 1:
                        start = int(pow(2, lev - 1)) - 1
                        end = int(pow(2, lev) - 1) - 1
                        for indx in range(start, end + 1):
                            self.LLRs[indx] = pcfun.upperconv(self.LLRs[end + 2 * (indx - start) + 1],
                                                              self.LLRs[end + 2 * (indx - start) + 2])
                    else:
                        lastlevel = (bin(pos_pre)[2:].zfill(self.n)).find('1') + 1
                        self.LLRs[0] = pcfun.lowerconv(self.BITS[0][0],
                                                       self.LLRs[1],
                                                       self.LLRs[2])
        else:
            self.LLRs[0] = pcfun.upperconv(self.LLRs[1],
                                             self.LLRs[2])

    def updateBITS_back(self, position: int, decoded: int):
        lastlevel = (bin(position)[2:].zfill(self.n)).find('0')
        bits_to_pass = int(np.exp2(lastlevel))
        i_last = pcfun.bitreversed(position, self.n)
        for i in range((i_last+1)-bits_to_pass, i_last+1):
            ii = pcfun.bitreversed(i, self.n)
            self.updateBITS(decoded[ii], ii)
                    
            
    def updateLLR_simplified(self, position: int, lastlevel: int):
        """updates LLR values at each step of SC decoding"""
        start = int(pow(2, lastlevel - 1)) - 1
        end = int(pow(2, lastlevel) - 1) - 1
        for i in range(start, end + 1):
            self.LLRs[i] = pcfun.lowerconv(self.BITS[0][i],
                                             self.LLRs[end + 2 * (i - start) + 1],
                                             self.LLRs[end + 2 * (i - start) + 2])

        if position == 0:
            nextlevel = self.n
        else:
            self.iter_clocks += 1
            lastlevel = (bin(position)[2:].zfill(self.n)).find('1') + 1
            start = int(pow(2, lastlevel - 1)) - 1
            end = int(pow(2, lastlevel) - 1) - 1
            for i in range(start, end + 1):
                self.LLRs[i] = pcfun.lowerconv(self.BITS[0][i],
                                               self.LLRs[end + 2 * (i - start) + 1],
                                               self.LLRs[end + 2 * (i - start) + 2])
                #print(self.BITS[0][i], self.LLRs[end + 2 * (i - start) + 1], self.LLRs[end + 2 * (i - start) + 2])
            nextlevel = lastlevel - 1
        #when 
        for lev in range(nextlevel, 0, -1):
            self.iter_clocks += 1
            start = int(pow(2, lev - 1)) - 1
            end = int(pow(2, lev) - 1) - 1
            for indx in range(start, end + 1):
                self.LLRs[indx] = pcfun.upperconv(self.LLRs[end + 2 * (indx - start) + 1],
                                                  self.LLRs[end + 2 * (indx - start) + 2])
                #print(self.LLRs[end + 2 * (indx - start) + 1], self.LLRs[end + 2 * (indx - start) + 2])

    def updateLLR_back1(self, pos_dest: int, pos_cur: int, decoded: int):
        #Assuming only one step has gone backward
        """updates LLR values at each step of SC decoding"""
        if pos_cur >= self.codeword_length//2:
            pos_cur = pcfun.bitreversed(pcfun.bitreversed(pos_cur, self.n) - 1, self.n)
        if pos_dest >= self.codeword_length//2: #in terms of i, they are odd, but in tems ii, they are >= N/2
            #The position of 1 from MSP tells us the starting stage of factor graph.
            #The starting stage includes g nodes.
            pos_dest_pre = pcfun.bitreversed(pcfun.bitreversed(pos_dest, self.n) - 1, self.n)
            #In natural order, ffs operator is used.    
            lastlevel_dest = (bin(pos_dest_pre)[2:].zfill(self.n)).find('1') + 1
            lastlevel_max = lastlevel_cur = (bin(pos_cur)[2:].zfill(self.n)).find('1') + 1

            i_dest_pre = pcfun.bitreversed(pos_dest_pre, self.n)
            i_cur = pcfun.bitreversed(pos_cur, self.n)
            for i in range(i_cur-1,i_dest_pre-1,-1):
                pos = pcfun.bitreversed(i, self.n)
                llevel = (bin(pos)[2:].zfill(self.n)).find('1') + 1
                lastlevel_max = llevel if llevel > lastlevel_max else lastlevel_max 

            if lastlevel_dest <= lastlevel_max:
                jmp = 3
                lastlevel_pre = lastlevel_dest
                while lastlevel_pre <= lastlevel_max:
                    pos_pre = pcfun.bitreversed(pcfun.bitreversed(pos_dest, self.n) - jmp, self.n)
                    if pos_pre > 0:
                        lastlevel_pre = (bin(pos_pre)[2:].zfill(self.n)).find('1') + 1
                    else:
                        lastlevel_pre = self.n
                        break
                    jmp += 2
                self.updateBITS_back(pos_pre, decoded)
                i_dest = pcfun.bitreversed(pos_dest, self.n)
                i_pre = pcfun.bitreversed(pos_pre, self.n)
                #New
                """ii = pcfun.bitreversed(i_pre, self.n)
                self.updateLLR(ii)
                for i in range(i_pre+1, i_dest+1): #I_dest updates are done in the main decoder
                    self.updateBITS(decoded[ii], ii)
                    ii = pcfun.bitreversed(i, self.n)"""
                for i in range(i_pre, i_dest+1):
                    ii = pcfun.bitreversed(i, self.n)
                    self.updateLLR(ii)
                    self.updateBITS(decoded[ii], ii)
                    
            elif  lastlevel_dest > lastlevel_max:
                if pos_dest_pre == 0:
                    nextlevel = self.n
                else:
                    self.iter_clocks += 1
                    #lastlevel = lastlevel_cur    
                    lastlevel = (bin(pos_dest_pre)[2:].zfill(self.n)).find('1') + 1
                    start = int(pow(2, lastlevel - 1)) - 1
                    end = int(pow(2, lastlevel) - 1) - 1
                    for i in range(start, end + 1):
                        self.total_additions += 1
                        self.LLRs[i] = pcfun.lowerconv(self.BITS[0][i],
                                                       self.LLRs[end + 2 * (i - start) + 1],
                                                       self.LLRs[end + 2 * (i - start) + 2])
                    nextlevel = lastlevel - 1
                #when bitreversal of position is >=N/2, because of lastlevel=1, the proceeding loop does not get through
                for lev in range(nextlevel, 0, -1):
                    self.iter_clocks += 1
                    if lev > 1:
                        start = int(pow(2, lev - 1)) - 1
                        end = int(pow(2, lev) - 1) - 1
                        for indx in range(start, end + 1):
                            self.total_comparisons += 1
                            self.LLRs[indx] = pcfun.upperconv(self.LLRs[end + 2 * (indx - start) + 1],
                                                              self.LLRs[end + 2 * (indx - start) + 2])
                    else:
                        lastlevel = (bin(pos_dest_pre)[2:].zfill(self.n)).find('1') + 1
                        self.total_additions += 1
                        self.LLRs[0] = pcfun.lowerconv(self.BITS[0][0],
                                                       self.LLRs[1],
                                                       self.LLRs[2])


        else:
            #when it is performed step by step (we always move from odd to even (0,2,4,...) bit index):
            self.LLRs[0] = pcfun.upperconv(self.LLRs[1],
                                             self.LLRs[2])
            #otherwise, jumping:
            #if pos_cur >= self.codeword_length//2:
                #pos_cur = pcfun.bitreversed(pcfun.bitreversed(pos_cur, self.n) - 1, self.n)
                
            lastlevel_dest = (bin(pos_dest)[2:].zfill(self.n)).find('1') + 1
            lastlevel_max = lastlevel_cur = (bin(pos_cur)[2:].zfill(self.n)).find('1') + 1

            i_dest = pcfun.bitreversed(pos_dest, self.n)
            i_cur = pcfun.bitreversed(pos_cur, self.n)
            for i in range(i_cur-1,i_dest-1,-1):
                pos = pcfun.bitreversed(i, self.n)
                llevel = (bin(pos)[2:].zfill(self.n)).find('1') + 1
                lastlevel_max = llevel if llevel > lastlevel_max else lastlevel_max 

            if lastlevel_dest <= lastlevel_max:
                jmp = 2
                lastlevel_pre = lastlevel_dest
                while lastlevel_pre <= lastlevel_max:
                    #print(pos_dest,self.n,jmp,lastlevel_pre,lastlevel_max)
                    pos_pre = pcfun.bitreversed(pcfun.bitreversed(pos_dest, self.n) - jmp, self.n)
                    if pos_pre > 0:
                        lastlevel_pre = (bin(pos_pre)[2:].zfill(self.n)).find('1') + 1
                    else:
                        lastlevel_pre = self.n
                        break
                    jmp += 2
                self.updateBITS_back(pos_pre, decoded)
                #i_dest = pcfun.bitreversed(pos_dest, self.n)
                i_pre = pcfun.bitreversed(pos_pre, self.n)
                #New
                #ii = pcfun.bitreversed(i_pre, self.n)
                """self.updateLLR(pos_pre)
                self.updateBITS(decoded[pos_pre], pos_pre)
                for i in range(i_pre+1, i_dest): #I_dest updates are done in the main decoder
                    ii = pcfun.bitreversed(i, self.n)
                    self.updateBITS(decoded[ii], ii)
                ii = pcfun.bitreversed(i_dest, self.n)
                self.updateLLR(ii)
                self.updateBITS(decoded[ii], ii)"""
                for i in range(i_pre, i_dest+1):
                    ii = pcfun.bitreversed(i, self.n)
                    self.updateLLR(ii)
                    self.updateBITS(decoded[ii], ii)
                    
            elif  lastlevel_dest > lastlevel_max:
                if pos_dest == 0:
                    nextlevel = self.n
                else:
                    self.iter_clocks += 1
                    #lastlevel = lastlevel_cur    
                    lastlevel = (bin(pos_dest)[2:].zfill(self.n)).find('1') + 1
                    start = int(pow(2, lastlevel - 1)) - 1
                    end = int(pow(2, lastlevel) - 1) - 1
                    for i in range(start, end + 1):
                        self.total_additions += 1
                        self.LLRs[i] = pcfun.lowerconv(self.BITS[0][i],
                                                       self.LLRs[end + 2 * (i - start) + 1],
                                                       self.LLRs[end + 2 * (i - start) + 2])
                    nextlevel = lastlevel - 1
                #when bitreversal of position is >=N/2, because of lastlevel=1, the proceeding loop does not get through
                for lev in range(nextlevel, 0, -1):
                    self.iter_clocks += 1
                    if lev > 1:
                        start = int(pow(2, lev - 1)) - 1
                        end = int(pow(2, lev) - 1) - 1
                        for indx in range(start, end + 1):
                            self.total_comparisons += 1
                            self.LLRs[indx] = pcfun.upperconv(self.LLRs[end + 2 * (indx - start) + 1],
                                                              self.LLRs[end + 2 * (indx - start) + 2])
                    else:
                        lastlevel = (bin(pos_dest)[2:].zfill(self.n)).find('1') + 1
                        self.total_additions += 1
                        self.LLRs[0] = pcfun.lowerconv(self.BITS[0][0],
                                                       self.LLRs[1],
                                                       self.LLRs[2])
            
                

    def move_back(self, pm_cur, bmetric, bmetric_cut, j, T, smaller_bm_followed, decoded, CS_flag):
        while True: #Look back
            follow_other_branch = 0
            mu_pre = bmetric[j-1] if j >= 1 else 0
            jj = j
            for k in range(j-1,-1,-1):
                #if CS_flag[k] == 1:
                mu_pre = bmetric[k]
                if mu_pre >= T:
                    if self.prnt_proc==1:
                        print("Looking back to j={0}.".format(k))
                    if bmetric_cut[k] >= T:
                        jj = k
                        tmp = bmetric[k]
                        bmetric[k] = bmetric_cut[k]
                        bmetric_cut[k] = tmp
                        if smaller_bm_followed[k] == 0:
                            #bmetric[k] = -100 #bmetric_cut[k] = -100 #tmp
                            follow_other_branch = 1
                            if self.prnt_proc==1:
                                print("Move back to j={0} and follow the worse branch.".format(k))
                            break
                        elif k == 0: #when worst node of root was followed, it was to start moving forward from root.
                            follow_other_branch = 1
                            jj = 0
                            break
                if k == 0:
                    mu_pre = -100
                    #jj = 0
                    if self.prnt_proc==1:
                        print("j={0}, None of the previous nodes satisfied the condition required for moving back.".format(k))

                
            i_cur = self.A[j] #output of self.A[j] is based on i
            #isFrozen_pre = 1 - self.polarcode_mask[pcfun.bitreversed(i_cur - 1, self.n)]
            if (mu_pre >= T) and (jj != 0 or follow_other_branch == 1): #or isFrozen_pre) and j != 0: #with j != 0, it cannot try 
                #jj = jj-1 if jj>=2 else 0  #move back one non-frozen node, when j==1 ==> j=0
                i_dest = self.A[jj]
                if self.prnt_proc==1 or self.prnt_proc==3:
                    print("{0}-->{1}, T={2}, pm_cur={4:0.3f} mu_pre={3:0.3f}".format(i_cur,i_dest, T, mu_pre, pm_cur))
                ii_dest = pcfun.bitreversed(i_dest, self.n) #considering the fz bits
                ii_cur = pcfun.bitreversed(i_cur, self.n)
                self.updateLLR_back1(ii_dest, ii_cur,decoded)
                #while i_cur > i_dest:
                    #i_cur -= 1
                    #ii_prepre = pcfun.bitreversed(i_cur-1, self.n) #considering the fz bits
                    #self.updateBITS(decoded[ii_prepre], ii_prepre)
                    #print(self.BITS)
                    #ii_pre = pcfun.bitreversed(i_cur, self.n) #not considering the fz bits
                    #self.updateLLR_back(ii_pre)
                    #print(self.LLRs[0])"""
                #print(j)
                #When it reaches to j==0, it falls into infinite loop here.
                
                if smaller_bm_followed[jj] == 0: #j+1: falls in a loop (goes back and forth b/w two nodes) #If the condition is not true, move back one more node by going thru next cycle of the infinite loop.
                    visited_before = 1
                    #self.updateBITS(decoded[ii1], ii1)
                    if self.prnt_proc==1 or self.prnt_proc==3:
                        print("************Move back & try the worst node************")
                    #We can store the metric of worst node in order not to recalculate it
                    return [T, jj, visited_before]
                elif jj == 0: #j+1: falls in a loop (goes back and forth b/w two nodes) #If the condition is not true, move back one more node by going thru next cycle of the infinite loop.
                    ##while pm_cur < T: #removing it might improve the perf
                    T = T - self.Delta
                    visited_before = 0
                    #self.updateBITS(decoded[ii1], ii1)
                    if self.prnt_proc==1 or self.prnt_proc==3:
                        print("************Move back to [info] root & follow the best node again************")
                    #We can store the metric of worst node in order not to recalculate it
                    return [T, jj, visited_before]
            else:   #Proceed the decoding with new T
                #while pm_cur < T: 
                T = T - self.Delta
                visited_before = 0
                #self.updateBITS(decoded[ii1], ii1)
                if self.prnt_proc==1:
                    print("**********Looked back**T reduced**********")
                return [T, j, visited_before]


    def move_back_stem(self, pm_cur, bmetric, bmetric_cut, j, T, smaller_bm_followed, decoded, CS_flag, mu_max_stem):
        visit_worst_stem = 0
        while True: #Look back
            follow_other_branch = 0
            mu_pre_worst = bmetric_cut[j-1] if j >= 1 else 0
            jj = j
            for k in range(j-1,-1,-1):
                if CS_flag[k] == 1:
                    mu_pre_worst = bmetric_cut[k]
                    """if mu_pre >= T:
                        if self.prnt_proc==1:
                            print("Looking back to j={0}.".format(k))"""
                    if bmetric_cut[k] >= T:
                        if self.prnt_proc==1:
                            print("Looking at j={0}.".format(k))
                        jj = k
                        tmp = bmetric[k]
                        bmetric[k] = bmetric_cut[k]
                        bmetric_cut[k] = tmp
                        if smaller_bm_followed[k] == 0:
                            #bmetric[k] = -100 #bmetric_cut[k] = -100 #tmp
                            follow_other_branch = 1
                            if self.prnt_proc==1:
                                print("Move into subtree j={0} by following the worse branch.".format(k))
                            break
                        elif k == 0: #when worst node of root was followed, it was to start moving forward from root.
                            follow_other_branch = 1
                            jj = 0
                            break
                    if k == 0:
                        mu_pre_worst = -100
                        #jj = 0
                        if self.prnt_proc==1:
                            print("j={0}, None of the previous nodes satisfied the condition required for moving back.".format(k))

                
            i_cur = self.A[j] #output of self.A[j] is based on i
            #isFrozen_pre = 1 - self.polarcode_mask[pcfun.bitreversed(i_cur - 1, self.n)]
            if (mu_pre_worst >= T) and (jj != 0 or follow_other_branch == 1): #or isFrozen_pre) and j != 0: #with j != 0, it cannot try 
                #jj = jj-1 if jj>=2 else 0  #move back one non-frozen node, when j==1 ==> j=0
                i_dest = self.A[jj]
                if self.prnt_proc==1 or self.prnt_proc==3:
                    print("{0}-->{1}, T={2}, pm_cur={4:0.3f} mu_pre_worst={3:0.3f}".format(i_cur,i_dest, T, mu_pre_worst, pm_cur))
                ii_dest = pcfun.bitreversed(i_dest, self.n) #considering the fz bits
                ii_cur = pcfun.bitreversed(i_cur, self.n)
                self.updateLLR_back1(ii_dest, ii_cur,decoded)
                #while i_cur > i_dest:
                    #i_cur -= 1
                    #ii_prepre = pcfun.bitreversed(i_cur-1, self.n) #considering the fz bits
                    #self.updateBITS(decoded[ii_prepre], ii_prepre)
                    #print(self.BITS)
                    #ii_pre = pcfun.bitreversed(i_cur, self.n) #not considering the fz bits
                    #self.updateLLR_back(ii_pre)
                    #print(self.LLRs[0])"""
                #print(j)
                #When it reaches to j==0, it falls into infinite loop here.
                
                if smaller_bm_followed[jj] == 0: #j+1: falls in a loop (goes back and forth b/w two nodes) #If the condition is not true, move back one more node by going thru next cycle of the infinite loop.
                    visited_before = 1
                    #self.updateBITS(decoded[ii1], ii1)
                    if self.prnt_proc==1 or self.prnt_proc==3:
                        print("************Move back & try the worst node************")
                    #We can store the metric of worst node in order not to recalculate it
                    return [T, jj, visited_before, visit_worst_stem]
                elif jj == 0: #j+1: falls in a loop (goes back and forth b/w two nodes) #If the condition is not true, move back one more node by going thru next cycle of the infinite loop.
                    while pm_cur < T: 
                        T = T - self.Delta
                    visited_before = 0
                    #self.updateBITS(decoded[ii1], ii1)
                    if self.prnt_proc==1 or self.prnt_proc==3:
                        print("************Move back to [info] root & follow the best node again************")
                    #We can store the metric of worst node in order not to recalculate it
                    return [T, jj, visited_before, visit_worst_stem]
            else:   #Proceed the decoding with new T
                while pm_cur < T: 
                    T = T - self.Delta
                visited_before = 0
                #self.updateBITS(decoded[ii1], ii1)
                if self.prnt_proc==1:
                    print("**********Looked back**T reduced**********")
                return [T, j, visited_before, visit_worst_stem]




    def move_back_stem_topdown(self, pm_cur, bmetric, bmetric_cut, j, T, smaller_bm_followed, decoded, CS_flag, ml_path_end):
        is_moving_back = 0
        """
        sbm_idx = 0
        bit_to_follow_mu_min = 0
        while sbm_idx <= j:
            if smaller_bm_followed[sbm_idx] == 0 and CS_flag[sbm_idx] == 1:
                bit_to_follow_mu_min = sbm_idx
                break
            #elif sbm_idx == j:
                #curr_mu_min_explored_idx =                 
            sbm_idx += 1
        """        
        while True: #Look back
            #print("$")
            jj = j
            if ml_path_end == 1: #Top-down movement
                for k in range(j):
                    if CS_flag[k] == 1:
                        mu_pre_worst = bmetric_cut[k]
                        if bmetric_cut[k] >= T:
                            jj = k
                            #follow_other_branch = 1
                            is_moving_back = 1
                            self.flip_cnt = 1
                            self.ml_exploring_mu_min_idx = k
                            if self.prnt_proc==1 or self.prnt_proc==3:
                                print("#####Start Exploring: Move into ml_path subtree j={0} & follow mu_min#####".format(k))
                            break
                if jj == j:
                    #This code exist in the main loop of decoding.
                    """mu_pre_worst = -100
                    while pm_cur < T: 
                        T = T - self.Delta"""
                    visited_before = 0
                    if self.prnt_proc==1 or self.prnt_proc==3:
                        print("j={0}. Condition not satisfied for moving back.".format(k)) #T reduced to {1}".format(k, T))
                    return [T, j, visited_before]

                            
            else:   #Bottom-up movement
                #follow_other_branch = 0
                mu_pre_worst = bmetric_cut[j-1] if j >= 1 else 0
                for k in range(j-1,-1,-1):
                    if self.ml_exploring_mu_min_idx == k: #Return to ml_path
                        jj = k
                        visited_before = 0
                        if self.prnt_proc==1 or self.prnt_proc==3:
                            print("#####Move back into ml_path & follow mu_max at j={0} ####.".format(k))
                        #if k != 0:
                        self.LLRs = copy.deepcopy(self.stem_LLRs)
                        self.BITS = copy.deepcopy(self.stem_BITS)
                        return [T, jj, visited_before]
                        #is_moving_back = 1
                        #break
                    if CS_flag[k] == 1:
                        mu_pre_worst = bmetric_cut[k]
                        """if mu_pre >= T:
                            if self.prnt_proc==1:
                                print("Looking back to j={0}.".format(k))"""
                        if bmetric_cut[k] >= T:
                            if self.prnt_proc==1:
                                print("Looking at j={0}.".format(k))
                            if sum(smaller_bm_followed[:k]) >= self.flips_const:
                                #print("Exceeded # of Flips at j={0}.".format(k))
                                continue
                            if smaller_bm_followed[k] == 0:
                                tmp = bmetric[k]
                                bmetric[k] = bmetric_cut[k]
                                bmetric_cut[k] = tmp
                                jj = k
                                #bmetric[k] = -100 #bmetric_cut[k] = -100 #tmp
                                #follow_other_branch = 1
                                is_moving_back = 1
                                self.flip_cnt += 1
                                if self.prnt_proc==1 or self.prnt_proc==3:
                                    print("******Move into subtree j={0} & follow mu_min,".format(k))
                                break
                            #In the above, this condition is satisied and the code doesn't reach here.
                            """elif k == ml_exloring_mu_min_idx: #when worst node of root was followed, it was to start moving forward from root.
                                follow_other_branch = 1
                                jj =  ml_exloring_mu_min_idx
                                break"""
                        """if k == 0:
                            mu_pre_worst = -100
                            #jj = 0
                            if self.prnt_proc==1:
                                print("j={0}, None of the previous nodes satisfied the condition required for moving back.".format(k))
                        """
                
            i_cur = self.A[j] #output of self.A[j] is based on i
            #isFrozen_pre = 1 - self.polarcode_mask[pcfun.bitreversed(i_cur - 1, self.n)]
            if is_moving_back == 1:
            #if (mu_pre_worst >= T) and (jj != ml_exloring_mu_min_idx or follow_other_branch == 1): #or isFrozen_pre) and j != 0: #with j != 0, it cannot try 
                #jj = jj-1 if jj>=2 else 0  #move back one non-frozen node, when j==1 ==> j=0
                i_dest = self.A[jj]
                if self.prnt_proc==1 or self.prnt_proc==3:
                    print("{0}-->{1}, T={2}, pm_cur={4:0.3f} mu_min_pre={3:0.3f}, ml_subtree j={5}".format(i_cur,i_dest, T, mu_pre_worst, pm_cur,self.ml_exploring_mu_min_idx))
                ii_dest = pcfun.bitreversed(i_dest, self.n) #considering the fz bits
                ii_cur = pcfun.bitreversed(i_cur, self.n)
                self.updateLLR_back1(ii_dest, ii_cur,decoded)
                
                if ml_path_end == 1:
                    self.stem_LLRs = copy.deepcopy(self.LLRs)
                    self.stem_BITS = copy.deepcopy(self.BITS)
                #while i_cur > i_dest:
                    #i_cur -= 1
                    #ii_prepre = pcfun.bitreversed(i_cur-1, self.n) #considering the fz bits
                    #self.updateBITS(decoded[ii_prepre], ii_prepre)
                    #print(self.BITS)
                    #ii_pre = pcfun.bitreversed(i_cur, self.n) #not considering the fz bits
                    #self.updateLLR_back(ii_pre)
                    #print(self.LLRs[0])"""
                #print(j)
                #When it reaches to j==0, it falls into infinite loop here.
                
                if smaller_bm_followed[jj] == 0: #j+1: falls in a loop (goes back and forth b/w two nodes) #If the condition is not true, move back one more node by going thru next cycle of the infinite loop.
                    visited_before = 1
                    #self.updateBITS(decoded[ii1], ii1)
                    #if self.prnt_proc==1 or self.prnt_proc==3:
                        #print("************Move back & try mu_min branch************")
                    #We can store the metric of worst node in order not to recalculate it
                    return [T, jj, visited_before]
                elif jj==0: #j+1: falls in a loop (goes back and forth b/w two nodes) #If the condition is not true, move back one more node by going thru next cycle of the infinite loop.
                    visited_before = 0
                    #self.updateBITS(decoded[ii1], ii1)
                    #if self.prnt_proc==1 or self.prnt_proc==3:
                        #print("####Move back to ml_path root & follow mu_max####")
                    #We can store the metric of worst node in order not to recalculate it
                    return [T, jj, visited_before]
            """else:   #Proceed the decoding with new T
                while pm_cur < T: 
                    T = T - self.Delta
                visited_before = 0
                #self.updateBITS(decoded[ii1], ii1)
                if self.prnt_proc==1:
                    print("**********Looked back**T reduced**********")
                return [T, j, visited_before]"""



    
    def PACfano_stem_topdown_decode2(self, soft_mess, issystematic: bool):
        mu_pre = -1000  #-infinity when mu_cur is the root node
        #mu_cur = 0
        #mu_look= 0
        #from_child_node = 0 #0: previous node was parent node, 1: was child node
        bmetric = [0.0 for i in range(self.information_size)]
        bmetric_cut = [0.0 for i in range(self.information_size)]
        bmetric_cut_updated = [-250.0 for i in range(self.information_size)]
        pm = [0 for i in range(self.codeword_length)]
        #bm_fz_pre = 0
        smaller_bm_followed = [0 for i in range(self.information_size)]
        smaller_bm_followed_on_stem = [0 for i in range(self.information_size)]
        visited_before = 0
        m = [0, 0]
        # reset LLRs
        self.LLRs = [0 for i in range(2 * self.codeword_length - 1)]
        # reset BITS
        self.BITS = [[0 for i in range(self.codeword_length - 1)] for j in range(2)]
        # reset decoding results
        decoded = [0 for i in range(self.codeword_length)]
        cdecoded = [0 for i in range(self.codeword_length)]
        # initial LLRs
        self.LLRs[self.codeword_length - 1:] = soft_mess
        frozen_set = (self.polarcode_mask + 1) % 2  #in bitrevesal order
        critical_set = pcfun.generate_critical_set(frozen_set)
        CS_flag = [0 for i in range(self.information_size)]
        top_ml_unexplored_idx = critical_set[0]
        top_unexplored_idx = 0
        next_idx_to_explore = 0
        subtree_bmetric_updated = 0
        ml_exploring_mu_min_idx = 0
        is_ml_exploring_mu_min = 0
        ml_idx_before_exploring = 0
        self.ml_exploring_mu_min_idx = 0
        encoded = [0, 0]
        branchState1 = [[],[]]
        branchState2 = [[],[]]
        state_cur1 = [[] for i in range(self.information_size)]
        state_cur2 = [[] for i in range(self.information_size)]
        i = 0
        i_pre = -1
        j = 0
        j_cs = 0
        B = 0
        B_delta = 0.5
        alpha = 1
        on_ml_path = 1
        ml_path_end = 0
        restart = 0
        tot_err_corrected = 0
        #stack
        seg_border = [39, 71, 127]
        seg_idx = 0
        bit_idx_B_updating = 39
        Bc = [0.0 for i in range(self.information_size)]
        for k in range(self.codeword_length):
            B += alpha*np.log(1-self.pe[k])
        Bcmplt = B
        j1=0
        for k in range(self.codeword_length):
            ii0 = pcfun.bitreversed(k, self.n)
            Bcmplt -= alpha*np.log(1-self.pe[ii0])
            if pcfun.bitreversed(self.A[0] - 1, self.n) == ii0:
                Bc0 = Bcmplt
            if pcfun.bitreversed(bit_idx_B_updating - 1, self.n) == ii0:
                Bc1 = Bcmplt
            if self.rate_profile[k] == 1:
                Bc[j1] = Bcmplt
                j1+=1
        """for k in range(self.codeword_length):
            Bcmplt -= alpha*np.log(1-self.pe[k])
            if self.A[0] - 1 == k:
                Bc0 = Bcmplt"""
        pm[-1]+=B
        if self.prnt_proc==1:
            print("B={0}".format(B))
        mu_max_stem = 0
        visit_worst_idx = 0
        bias_updated = 0
        while i < self.codeword_length:
            """if i > seg_border[0] and i <= seg_border[1] and on_ml_path == 1:
                seg_idx = 1
            elif i <= seg_border[0] and on_ml_path == 1:
                seg_idx = 0
            elif i > seg_border[1] and on_ml_path == 1:
                seg_idx = 2"""
                    
            """if self.iter_clocks>254*256 and restart == 0: #No significabt effect on clocks_ave
                self.T = -1000
                j = 0
                i  = 0
                restart = 1"""
                #print("Clocks exceeded the cap")"""
            if i != i_pre: #or self.ml_exploring_mu_min_idx !=j: #to avoid recalculation after reucing T in the begining
                i_pre = i
                ii = pcfun.bitreversed(i, self.n)
                
                self.updateLLR(ii)
                dLLR = self.LLRs[0]
                abs_dLLR = np.abs(dLLR)
                self.total_steps += 1
            #print(ii)
            #if self.total_clocks>self.max_clocks:
                #self.T = -100
            p_y = np.log(0.5/(np.sqrt(2*3.14)*self.sigma)*(np.exp(-np.square(soft_mess[ii]-1)/(2*np.square(self.sigma)))+np.exp(-np.square(soft_mess[ii]+1)/(2*np.square(self.sigma)))))
            #print(soft_mess[ii], self.sigma, p_y)
            if self.polarcode_mask[ii] == 1:
                #if j==0 and : 
                #jj = 0 if j==0 else j-1
                ##mu_pre = 0 if j == 0 else bmetric[j-1] #mu_pre = bmetric[jj]
                #Calc mu_look:
                encoded[0], branchState1[0], branchState2[0] =  pcfun.conv1bit_getNextStates(0, self.cur_state1, self.cur_state2, self.gen1, self.gen2, self.critical_set_flag[i])
                encoded[1], branchState1[1], branchState2[1] =  pcfun.conv1bit_getNextStates(1, self.cur_state1, self.cur_state2, self.gen1, self.gen2, self.critical_set_flag[i])
                """encoded[0] = pcfun.conv_1bit(0, self.cur_state, self.gen)
                branchState[0] = pcfun.getNextState(0, self.cur_state, self.m)
                encoded[1] = 1 - encoded[0]
                branchState[1] = pcfun.getNextState(1, self.cur_state, self.m)"""
                #@3.5dB, p_max: RuntimeWarning: invalid value encountered in double_scalars
                if (abs_dLLR>600): #abs_dLLR.astype(np.float128)
                    p_max = 1
                else:
                    p_max = np.exp(abs_dLLR)/(np.exp(abs_dLLR)+1)
                p_min = 1-p_max #1/(np.exp(abs_dLLR)+1)
                #when you try to evaluate log with 0 (not dividng by zero):
                #RuntimeWarning: divide by zero encountered in log
                b = alpha*np.log(1-self.pe[ii])
                p_min += 10**-7
                if dLLR > 0: #p_u0>p_u1
                    m[0] = pm[i-1] + np.log((p_max if encoded[0]==0 else p_min)) - b
                    m[1] = pm[i-1] + np.log((p_max if encoded[1]==0 else p_min)) - b
                    #m[0] = pm[i-1] + np.log(p_max if encoded[0]==0 else p_min)
                    #m[1] = pm[i-1] + np.log(p_max if encoded[1]==0 else p_min)
                    if self.prnt_proc==1:
                        print("{4}: dllr={0:0.2f}, Pu0={1:0.2f} Pu1={2:0.2f}, 1-Pe={3:0.5f}".format(dLLR,(np.exp(dLLR)/(np.exp(dLLR)+1)),(1/(np.exp(dLLR)+1)),(1-self.pe[ii]),i))
                else:   #p_u1>p_u0
                    m[0] = pm[i-1] + np.log((p_max if encoded[0]==1 else p_min)) - b
                    m[1] = pm[i-1] + np.log((p_max if encoded[1]==1 else p_min)) - b
                    #m[0] = pm[i-1] + np.log(p_max if encoded[0]==1 else p_min)
                    #m[1] = pm[i-1] + np.log(p_max if encoded[1]==1 else p_min)
                    if self.prnt_proc==1:
                        print("{4}: dllr={0:0.2f}, Pu0={2:0.2f} Pu1={1:0.2f}, 1-Pe={3:0.5f}".format(dLLR,(np.exp(dLLR)/(np.exp(dLLR)+1)),(1/(np.exp(dLLR)+1)),(1-self.pe[ii]),i))
                mu_max = m[0] if m[0] > m[1] else m[1]    #best mu_look
                mu_min = m[0] if m[0] < m[1] else m[1]    #next best mu_look
                #the commented part: to expand subtrees segment-wise >> no sig. change in complx
                #introduces a slight error despite slight reduction in cplx
                if on_ml_path == 1 and is_ml_exploring_mu_min == 1:
                    if CS_flag[j] == 1 and ml_idx_before_exploring > j and self.ml_exploring_mu_min_idx < j and mu_min > self.T: #smaller_bm_followed[j] == 0:
                        visited_before = 1
                        self.flip_cnt = 1 #sum is being used so this line is not relevant
                        on_ml_path = 0
                        self.stem_LLRs = copy.deepcopy(self.LLRs)
                        self.stem_BITS = copy.deepcopy(self.BITS)
                        self.ml_exploring_mu_min_idx = j
                        smaller_bm_followed_on_stem[j] = 1
                        if self.prnt_proc==1 or self.prnt_proc==3:
                            print("#####Moving into subtree j={0} from ml_path, last ml bit j={1}#####".format(j,ml_idx_before_exploring))
                    if ml_idx_before_exploring ==  j: #Could happen after exploring or when condition not satisfied.
                        is_ml_exploring_mu_min = 0
                        while self.ml_last_mu_max < self.T: 
                            self.T = self.T - self.Delta
                        if self.prnt_proc==1 or self.prnt_proc==3:
                            print("#####End of exploring subtrees. Proceed with ml_path from j={0},i={1}#####".format(j,i))
                            print("T reduced to {0}, mu_max(u{1}):{2:0.2f} < T".format(self.T,i,self.ml_last_mu_max))
                
                if mu_max >= self.T: #or (mu_max >= (self.T-self.Delta) and i < seg_border[seg_idx] and on_ml_path != 1):
                    #It could be moving forward after backward movement(s):
                    if visited_before == 0:
                        #Move forward:
                        #From now on, mu_look --> mu_cur
                        cdecoded[ii] = 0 if m[0]>m[1] else 1
                        #decoded[ii] = encoded[0] if m[0]>m[1] else encoded[1]
                        decoded[ii] = encoded[cdecoded[ii]]
                        """if bmetric[j] < bmetric_cut[j] and smaller_bm_followed[j] == 1: #They may have been swapped when exploring worst node..
                            tmp = bmetric[j]
                            bmetric[j] = bmetric_cut[j]
                            bmetric_cut[j] = tmp"""
                        bmetric[j] = mu_max #mu_cur (after forward move), previously mu_look
                        #bmetric_cut[j] = mu_min
                        #updated worst node PM
                        bmetric_cut[j] = bmetric_cut_updated[j] if (on_ml_path == 1 and smaller_bm_followed_on_stem[j] == 1) else mu_min #self.ml_exploring_mu_min_idx == j) else mu_min
                        pm[i] = mu_max
                        #bm_fz_pre = 0
                        smaller_bm_followed[j] = 0
                        #Ginie corrects the decoded bits, when T is very small:
                        if self.prnt_proc==4:
                            if self.trdata[ii] != cdecoded[ii]:
                                cdecoded[ii] = 1 - cdecoded[ii]
                                decoded[ii] = encoded[cdecoded[ii]]
                                bmetric[j] = mu_min #mu_cur (after forward move), previously mu_look
                                bmetric_cut[j] = mu_max
                                pm[i] = mu_min
                                smaller_bm_followed[j] = 1
                                #Statistics:
                                self.bit_err_cnt[i] += 1
                                tot_err_corrected += 1
                        self.updateBITS(decoded[ii], ii)
                        #print(j)

                        state_cur1[j] = self.cur_state1    #Current state
                        state_cur2[j] = self.cur_state2    #Current state
                        self.cur_state1 = branchState1[cdecoded[ii]]  #Next state
                        self.cur_state2 = branchState2[cdecoded[ii]]  #Next state
                        """state_cur[j] = self.cur_state    #Current state
                        self.cur_state = branchState[cdecoded[ii]]  #Next state"""
                        if self.prnt_proc==1:
                            print("v={5}, u={4}: max({2:0.3f},{3:0.3f}) > T={1}, bm_cut={10:0.2f}, nState({8})={7}, pm={6:0.3f}, Clks={9}".format(i,self.T,m[0],m[1],decoded[ii],cdecoded[ii],mu_max,self.cur_state,j,self.total_clocks,bmetric_cut[j]))
                        if critical_set[j_cs] == i and CS_flag[j] != 1:
                            CS_flag[j] = 1
                            if j_cs < critical_set.size - 1:
                                j_cs += 1
                        #mu_pre = 0 if j == 0 else bmetric[j-1]#line453 does the same but more accurate for j=0
                        mu_pre = pm[i-1]
                        #T = mu_max
                        #Tightening the threshold in the first visit: Two conditionsfor increasing T
                        """if mu_pre < T + self.Delta: #i.e. n_pre cannot have been visited with higher threshold: previous visit of n_pre was the first visit: n_cur is also visited for the first time.
                            while bmetric[j] > T + self.Delta: #When T is too low; several cycles
                                T = T + self.Delta"""
                        i += 1
                        j += 1
                    else:
                        #mu_min = m[0] if m[0] < m[1] else m[1]    #next best mu_look
                        #print("mu_min={0} > {1} ? mu_max={2}".format(mu_min, self.T, mu_max))
                        if mu_min > self.T or mu_min <= self.T: #This condition is not required
                            cdecoded[ii] = 0 if m[0]<m[1] else 1
                            decoded[ii] = encoded[0] if m[0]<m[1] else encoded[1]
                            self.updateBITS(decoded[ii], ii)
                            bmetric[j] = mu_min #mu_cur
                            bmetric_cut[j] = mu_max #if bmetric_cut[j] > mu_max else bmetric_cut[j]
                            pm[i] = mu_min
                            state_cur1[j] = self.cur_state1    #Current state
                            state_cur2[j] = self.cur_state2    #Current state
                            self.cur_state1 = branchState1[cdecoded[ii]]  #Next state
                            self.cur_state2 = branchState2[cdecoded[ii]]  #Next state
                            """state_cur[j] = self.cur_state 
                            self.cur_state = branchState[cdecoded[ii]]"""
                            if m[0]==m[1]: #They might get very close but it never happens
                                print("m0==m1?")
                            if self.prnt_proc==1 or (self.prnt_proc==3 and on_ml_path == 0 and self.ml_exploring_mu_min_idx == j):
                                print("v={5}, u={4}: min({2:0.6f}, {3:0.6f}) > T={1}, bm_cut={9:0.2f}, nState({8})={7}, pm={6:0.3f}".format(i,self.T,m[0],m[1],decoded[ii],cdecoded[ii],mu_min,self.cur_state,j,bmetric_cut[j]))
                            #bm_fz_pre = 0
                            smaller_bm_followed[j] = 1
                            i += 1
                            j += 1
                            visited_before = 0
                        else:   #This part is not used because there is code in move_back the does the same thing.
                            #Segment-wise
                            print("******************************************************************************************************************************************")
                            """if j == 0 or (on_ml_path == 1 and (i<=31 or (i<=63 and i>39))):
                            #if j == 0:
                                self.T = self.T - self.Delta
                                visited_before = 0
                                if self.prnt_proc==1:
                                    print("T: {0} => {1}, mu_min(u{2}) < T, Move FWD to the best node".format(self.T + self.Delta,self.T,i))
                            else:
                                if self.prnt_proc==1:
                                    print("u{0}: min({1:0.3f}, {2:0.3f}) < T:{3}, cState({5})={4}, Look BACK".format(i,m[0],m[1],self.T,self.cur_state,j))
                                state_cur[j] = self.cur_state
                                
                                [self.T, j, visited_before, visit_worst_idx] = self.move_back_stem(mu_min, bmetric, bmetric_cut, j, self.T, smaller_bm_followed, decoded, CS_flag, mu_max_stem)
                                self.cur_state = [0 for mm in range(self.m)] if j==0 else state_cur[j]
                                if visited_before == 0 and j==0: 
                                    on_ml_path = 1
                                else:
                                    on_ml_path = 0
                                if self.prnt_proc==1:
                                    print("cState({1})={0}".format(self.cur_state,j))
                                i = self.A[j]
                                #bm_fz_pre = 0
                                #i = pcfun.bitreversed(ii1, self.n)"""

                else:
                    #Segment-wise
                    #if j == 0 or (on_ml_path == 1 and ( i<=31 or (i<=63 and i>39) )):#i<=31 or (i<=63 and i>39)  or (i<=95 and i>71)
                    if j == 0:
                        while mu_max < self.T: #Adjusting the threshold in the begning
                            self.T = self.T - self.Delta
                        visited_before = 0
                        if self.prnt_proc==1:
                            print("$$$$ T reduced to {0}, mu_max(u{1}) < T, Move FWD to the best node".format(self.T,i))
                    else:
                        if self.prnt_proc==1:
                            print("u{0}: max({1:0.3f}, {2:0.3f}) < T:{3}, cState({5})={4}, Look BACK".format(i,m[0],m[1],self.T,self.cur_state,j))
                        state_cur1[j] = self.cur_state1    #Current state
                        state_cur2[j] = self.cur_state2    #Current state
                        #state_cur[j] = self.cur_state
                        #Reduce treshold down to mu_max to avoid several cycles
                        """if on_ml_path == 1 and (i==39 or i==71):# 
                            while mu_max < self.T - self.Delta: #Adjusting the threshold in the begning
                                self.T = self.T - self.Delta"""
                        #if on_ml_path == 1:
                            #ml_down_explored_idx = 
                        #updated worst node PM   #bmetric_cut[j] should change as well
                        """for k in range(j):#self.information_size): #reduced clck_avg by introducing error
                            if smaller_bm_followed[k] == 1:
                                #self.ml_exploring_mu_min_idx = k #
                                smaller_bm_followed_on_stem[k] = 1
                                #next_ml_idx_to_explore = i
                                if bmetric_cut_updated[k] < mu_max:
                                     bmetric_cut_updated[k] = mu_max
                                     if self.prnt_proc==1 or self.prnt_proc==3:
                                        print("$$$$$$$ Updating bmetric_cut[{0}] = {1:0.2f}************".format(k,mu_max))
                                break"""
                        #It does the same as the above.
                        #"""
                        if on_ml_path == 0:
                            if bmetric_cut_updated[self.ml_exploring_mu_min_idx] < mu_max:
                                bmetric_cut_updated[self.ml_exploring_mu_min_idx] = mu_max
                                if self.prnt_proc==1 or self.prnt_proc==3:
                                    print("$$$$$$$ Updating bmetric_cut[{0}] = {1:0.2f}************".format(self.ml_exploring_mu_min_idx,mu_max))
                        #"""      
                        #Updating biases based on the noise
                        if bias_updated == 0 and on_ml_path == 1 and i == bit_idx_B_updating: #Extra condition are essential
                            if mu_max < B:
                                alpha0 = np.ceil(np.abs(mu_max/B)/B_delta)*B_delta
                                alpha = alpha0 #if alpha0 > ?
                                if self.prnt_proc==1 or self.prnt_proc==3:
                                    print("&&&&&&&& Updating alpha={0:0.2f}, B={1:0.2f} < mu={2:0.2f}".format(alpha,B,mu_max))
                            bias_updated = 1
                            for j2 in range(j):
                                bmetric_cut[j2] = bmetric_cut[j2] + (alpha-1)*Bc[j2]
                            pm[self.A[0]-1] = pm[self.A[0]-1] + (alpha-1)*Bc0
                            pm[i-1] = pm[i-1] + (alpha-1)*Bc1
                            pm[i] = pm[i-1] + np.log(p_max) - alpha*np.log(1-self.pe[ii]) #mu_max
                            #mu_max = pm[i]
                            #"""
                        if on_ml_path == 1:
                            ml_idx_before_exploring = j
                            is_ml_exploring_mu_min = 1
                            ml_path_end = 1
                            self.ml_last_mu_max = mu_max
                            if self.prnt_proc==1 or self.prnt_proc==3:
                                print("****Examining subtrees of ml_path, last ml bit j={0},i={1}****".format(j,i))
                            #mu_max_stem = mu_max
                            
                        [self.T, j1, visited_before] = self.move_back_stem_topdown(mu_max, bmetric, bmetric_cut, j, self.T, smaller_bm_followed, decoded, CS_flag, ml_path_end)
                        if visited_before == 0 and (j1==self.ml_exploring_mu_min_idx or j1==j): #this part may switch to = 0 wrongly when it proceeds to decoding w/o going to origin
                            on_ml_path = 1
                            #if self.prnt_proc==1 or self.prnt_proc==3:
                                #print("Entering ml_path")
                        else:
                            on_ml_path = 0
                            #if self.prnt_proc==1 or self.prnt_proc==3:
                                #print("Leaving ml_path")
                        j=j1
                        ml_path_end = 0
                        self.cur_state1 = [0 for mm in range(self.m1)] if j==0 else state_cur1[j]
                        self.cur_state2 = [0 for mm in range(self.m2)] if j==0 else state_cur2[j]
                        #self.cur_state = [0 for mm in range(self.m)] if j==0 else state_cur[j]
                        if self.prnt_proc==1:
                            print("cState({1})={0}".format(self.cur_state,j))
                        i = self.A[j]
                        #bm_fz_pre = 0
                        #i = pcfun.bitreversed(ii1, self.n)
                    
            else:
                #if dLLR < 0:
                #if j>0:
                    #bmetric[0 if j==0 else j-1] += np.log((np.exp(dLLR)/(np.exp(dLLR)+1))/(1-self.pe[ii]))
                ##bm_fz_pre += np.log((np.exp(dLLR)/(np.exp(dLLR)+1))/(1-self.pe[ii]))
                #encoded[0], branchState1[0], branchState2[0] =  pcfun.conv1bit_getNextStates(0, self.cur_state1, self.cur_state2, self.gen1, self.gen2, self.critical_set_flag[i])
                encoded[0] = pcfun.conv_1bit2(0, self.cur_state1, self.cur_state2, self.gen1, self.gen2, self.critical_set_flag[j])
                #branchState1[0], branchState2[0] = pcfun.getNextState2(0, self.cur_state1, self.cur_state2, self.gen1, self.gen2, self.critical_set_flag[j])
                p_max = np.exp(abs_dLLR)/(np.exp(abs_dLLR)+1)
                p_min = 1/(np.exp(abs_dLLR)+1)
                #if on_ml_path == 1:
                    #p_min = p_min if p_min>0.1 else 0.1
                bm = np.log((p_max if (encoded[0]==0 and dLLR>0) or (encoded[0]==1 and dLLR<0) else p_min))
                #print(bm)
                #if on_ml_path == 1:
                    #bm = bm if bm>-2 else -2
                pm[i] = pm[i-1] + bm -alpha*np.log(1-self.pe[ii])
                #pm[i] = pm[i-1] + np.log(p_max if (encoded[0]==0 and dLLR>0) or (encoded[0]==1 and dLLR<0) else p_min) - 0.7
                self.cur_state1 = branchState1[0]  #Next state
                self.cur_state2 = branchState2[0]  #Next state
                state_cur1[j] = self.cur_state1    #Current state
                state_cur2[j] = self.cur_state2    #Current state
                if self.prnt_proc==1:
                    print("{3}: dllr={0:0.2f}, Pu0={1:0.2f}, 1-Pe={2:0.6f}".format(dLLR,(np.exp(dLLR)/(np.exp(dLLR)+1)),(1-self.pe[ii]),i))
                """else:
                    bmetric[0 if j==0 else j-1] += np.log(1-(np.exp(dLLR)/(np.exp(dLLR)+1))/(1-self.pe[ii]))"""

                cdecoded[ii] = 0
                decoded[ii] = encoded[0]
                if self.prnt_proc==1:
                    print("v{0}=FZ, u{0}={3}: T={1}, cState({5})={4}, pm={2:0.3f}".format(i,self.T,pm[i],encoded[0],self.cur_state,j))
                self.updateBITS(decoded[ii], ii)
                i += 1
            
        if self.prnt_proc==4:
            self.tot_err_freq[tot_err_corrected] += 1
        return self.extract(cdecoded)
        




    def PACfano_stem_topdown_decoder(self, soft_mess, issystematic: bool):
        mu_pre = -1000  #-infinity when mu_cur is the root node
        #mu_cur = 0
        #mu_look= 0
        #from_child_node = 0 #0: previous node was parent node, 1: was child node
        bmetric = [0.0 for i in range(self.information_size)]
        bmetric_cut = [0.0 for i in range(self.information_size)]
        bmetric_cut_updated = [-250.0 for i in range(self.information_size)]
        pm = [0 for i in range(self.codeword_length)]
        #bm_fz_pre = 0
        smaller_bm_followed = [0 for i in range(self.information_size)]
        smaller_bm_followed_on_stem = [0 for i in range(self.information_size)]
        visited_before = 0
        m = [0, 0]
        # reset LLRs
        self.LLRs = [0 for i in range(2 * self.codeword_length - 1)]
        # reset BITS
        self.BITS = [[0 for i in range(self.codeword_length - 1)] for j in range(2)]
        # reset decoding results
        decoded = [0 for i in range(self.codeword_length)]
        cdecoded = [0 for i in range(self.codeword_length)]
        # initial LLRs
        self.LLRs[self.codeword_length - 1:] = soft_mess
        frozen_set = (self.polarcode_mask + 1) % 2  #in bitrevesal order
        critical_set = pcfun.generate_critical_set(frozen_set)
        CS_flag = [0 for i in range(self.information_size)]
        top_ml_unexplored_idx = critical_set[0]
        top_unexplored_idx = 0
        next_idx_to_explore = 0
        subtree_bmetric_updated = 0
        ml_exploring_mu_min_idx = 0
        is_ml_exploring_mu_min = 0
        ml_idx_before_exploring = 0
        self.ml_exploring_mu_min_idx = 0
        encoded = [0, 0]
        branchState = [[],[]]
        state_cur = [[] for i in range(self.information_size)]
        i = 0
        i_pre = -1
        j = 0
        j_cs = 0
        B = 0
        B_delta = 0.5
        alpha = 1
        on_ml_path = 1
        ml_path_end = 0
        restart = 0
        tot_err_corrected = 0
        #stack
        seg_border = [39, 71, 127]
        seg_idx = 0
        bit_idx_B_updating = self.bit_idx_B_updating
        Bc = [0.0 for i in range(self.information_size)]
        for k in range(self.codeword_length):
            B += alpha*np.log(1-self.pe[k])
        Bcmplt = B
        j1=0
        for k in range(self.codeword_length):
            ii0 = pcfun.bitreversed(k, self.n)
            Bcmplt -= alpha*np.log(1-self.pe[ii0])
            if pcfun.bitreversed(self.A[0] - 1, self.n) == ii0:
                Bc0 = Bcmplt
            if pcfun.bitreversed(bit_idx_B_updating - 1, self.n) == ii0:
                Bc1 = Bcmplt
            if self.rate_profile[k] == 1:
                Bc[j1] = Bcmplt
                j1+=1
        """for k in range(self.codeword_length):
            Bcmplt -= alpha*np.log(1-self.pe[k])
            if self.A[0] - 1 == k:
                Bc0 = Bcmplt"""
        pm[-1]+=B
        if self.prnt_proc==1:
            print("B={0}".format(B))
        mu_max_stem = 0
        visit_worst_idx = 0
        bias_updated = 0
        while i < self.codeword_length:
            """if i > seg_border[0] and i <= seg_border[1] and on_ml_path == 1:
                seg_idx = 1
            elif i <= seg_border[0] and on_ml_path == 1:
                seg_idx = 0
            elif i > seg_border[1] and on_ml_path == 1:
                seg_idx = 2"""
                    
            """if self.iter_clocks>254*256 and restart == 0: #No significabt effect on clocks_ave
                self.T = -1000
                j = 0
                i  = 0
                restart = 1"""
                #print("Clocks exceeded the cap")"""
            if i != i_pre: #or self.ml_exploring_mu_min_idx !=j: #to avoid recalculation after reucing T in the begining
                i_pre = i
                ii = pcfun.bitreversed(i, self.n)
                
                self.updateLLR(ii)
                dLLR = self.LLRs[0]
                abs_dLLR = np.abs(dLLR)
                self.total_steps += 1
            #print(ii)
            #if self.total_clocks>self.max_clocks:
                #self.T = -100
            p_y = np.log(0.5/(np.sqrt(2*3.14)*self.sigma)*(np.exp(-np.square(soft_mess[ii]-1)/(2*np.square(self.sigma)))+np.exp(-np.square(soft_mess[ii]+1)/(2*np.square(self.sigma)))))
            #print(soft_mess[ii], self.sigma, p_y)
            if self.polarcode_mask[ii] == 1:
                #if j==0 and : 
                #jj = 0 if j==0 else j-1
                ##mu_pre = 0 if j == 0 else bmetric[j-1] #mu_pre = bmetric[jj]
                #Calc mu_look:
                encoded[0] = pcfun.conv_1bit(0, self.cur_state, self.gen)
                branchState[0] = pcfun.getNextState(0, self.cur_state, self.m)
                encoded[1] = 1 - encoded[0]
                branchState[1] = pcfun.getNextState(1, self.cur_state, self.m)
                #@3.5dB, p_max: RuntimeWarning: invalid value encountered in double_scalars
                p_max = np.exp(abs_dLLR)/(np.exp(abs_dLLR)+1)
                p_min = 1-p_max #1/(np.exp(abs_dLLR)+1)
                #when you try to evaluate log with 0 (not dividng by zero):
                #RuntimeWarning: divide by zero encountered in log
                b = alpha*np.log(1-self.pe[ii])
                p_min += 10**-7
                if dLLR > 0: #p_u0>p_u1
                    m[0] = pm[i-1] + np.log((p_max if encoded[0]==0 else p_min)) - b
                    m[1] = pm[i-1] + np.log((p_max if encoded[1]==0 else p_min)) - b
                    #m[0] = pm[i-1] + np.log(p_max if encoded[0]==0 else p_min)
                    #m[1] = pm[i-1] + np.log(p_max if encoded[1]==0 else p_min)
                    if self.prnt_proc==1:
                        print("{4}: dllr={0:0.2f}, Pu0={1:0.2f} Pu1={2:0.2f}, 1-Pe={3:0.5f}".format(dLLR,(np.exp(dLLR)/(np.exp(dLLR)+1)),(1/(np.exp(dLLR)+1)),(1-self.pe[ii]),i))
                else:   #p_u1>p_u0
                    m[0] = pm[i-1] + np.log((p_max if encoded[0]==1 else p_min)) - b
                    m[1] = pm[i-1] + np.log((p_max if encoded[1]==1 else p_min)) - b
                    #m[0] = pm[i-1] + np.log(p_max if encoded[0]==1 else p_min)
                    #m[1] = pm[i-1] + np.log(p_max if encoded[1]==1 else p_min)
                    if self.prnt_proc==1:
                        print("{4}: dllr={0:0.2f}, Pu0={2:0.2f} Pu1={1:0.2f}, 1-Pe={3:0.5f}".format(dLLR,(np.exp(dLLR)/(np.exp(dLLR)+1)),(1/(np.exp(dLLR)+1)),(1-self.pe[ii]),i))
                mu_max = m[0] if m[0] > m[1] else m[1]    #best mu_look
                mu_min = m[0] if m[0] < m[1] else m[1]    #next best mu_look
                #the commented part: to expand subtrees segment-wise >> no sig. change in complx
                #introduces a slight error despite slight reduction in cplx
                if on_ml_path == 1 and is_ml_exploring_mu_min == 1:
                    if CS_flag[j] == 1 and ml_idx_before_exploring > j and self.ml_exploring_mu_min_idx < j and mu_min > self.T: #smaller_bm_followed[j] == 0:
                        visited_before = 1
                        self.flip_cnt = 1 #sum is being used so this line is not relevant
                        on_ml_path = 0
                        self.stem_LLRs = copy.deepcopy(self.LLRs)
                        self.stem_BITS = copy.deepcopy(self.BITS)
                        self.ml_exploring_mu_min_idx = j
                        smaller_bm_followed_on_stem[j] = 1
                        if self.prnt_proc==1 or self.prnt_proc==3:
                            print("#####Moving into subtree j={0} from ml_path, last ml bit j={1}#####".format(j,ml_idx_before_exploring))
                    if ml_idx_before_exploring ==  j: #Could happen after exploring or when condition not satisfied.
                        is_ml_exploring_mu_min = 0
                        while self.ml_last_mu_max < self.T: 
                            self.T = self.T - self.Delta
                        if self.prnt_proc==1 or self.prnt_proc==3:
                            print("#####End of exploring subtrees. Proceed with ml_path from j={0},i={1}#####".format(j,i))
                            print("T reduced to {0}, mu_max(u{1}):{2:0.2f} < T".format(self.T,i,self.ml_last_mu_max))
                
                if mu_max >= self.T: #or (mu_max >= (self.T-self.Delta) and i < seg_border[seg_idx] and on_ml_path != 1):
                    #It could be moving forward after backward movement(s):
                    if visited_before == 0:
                        #Move forward:
                        #From now on, mu_look --> mu_cur
                        cdecoded[ii] = 0 if m[0]>m[1] else 1
                        #decoded[ii] = encoded[0] if m[0]>m[1] else encoded[1]
                        decoded[ii] = encoded[cdecoded[ii]]
                        """if bmetric[j] < bmetric_cut[j] and smaller_bm_followed[j] == 1: #They may have been swapped when exploring worst node..
                            tmp = bmetric[j]
                            bmetric[j] = bmetric_cut[j]
                            bmetric_cut[j] = tmp"""
                        bmetric[j] = mu_max #mu_cur (after forward move), previously mu_look
                        #bmetric_cut[j] = mu_min
                        #updated worst node PM
                        bmetric_cut[j] = bmetric_cut_updated[j] if (on_ml_path == 1 and smaller_bm_followed_on_stem[j] == 1) else mu_min #self.ml_exploring_mu_min_idx == j) else mu_min
                        pm[i] = mu_max
                        #bm_fz_pre = 0
                        smaller_bm_followed[j] = 0
                        #Ginie corrects the decoded bits, when T is very small:
                        if self.prnt_proc==4:
                            if self.trdata[ii] != cdecoded[ii]:
                                cdecoded[ii] = 1 - cdecoded[ii]
                                decoded[ii] = encoded[cdecoded[ii]]
                                bmetric[j] = mu_min #mu_cur (after forward move), previously mu_look
                                bmetric_cut[j] = mu_max
                                pm[i] = mu_min
                                smaller_bm_followed[j] = 1
                                #Statistics:
                                self.bit_err_cnt[i] += 1
                                tot_err_corrected += 1
                        self.updateBITS(decoded[ii], ii)
                        #print(j)

                        state_cur[j] = self.cur_state    #Current state
                        self.cur_state = branchState[cdecoded[ii]]  #Next state
                        if self.prnt_proc==1:
                            print("v={5}, u={4}: max({2:0.3f},{3:0.3f}) > T={1}, bm_cut={10:0.2f}, nState({8})={7}, pm={6:0.3f}, Clks={9}".format(i,self.T,m[0],m[1],decoded[ii],cdecoded[ii],mu_max,self.cur_state,j,self.total_clocks,bmetric_cut[j]))
                        if critical_set[j_cs] == i and CS_flag[j] != 1:
                            CS_flag[j] = 1
                            if j_cs < critical_set.size - 1:
                                j_cs += 1
                        #mu_pre = 0 if j == 0 else bmetric[j-1]#line453 does the same but more accurate for j=0
                        mu_pre = pm[i-1]
                        #T = mu_max
                        #Tightening the threshold in the first visit: Two conditionsfor increasing T
                        """if mu_pre < T + self.Delta: #i.e. n_pre cannot have been visited with higher threshold: previous visit of n_pre was the first visit: n_cur is also visited for the first time.
                            while bmetric[j] > T + self.Delta: #When T is too low; several cycles
                                T = T + self.Delta"""
                        i += 1
                        j += 1
                    else:
                        #mu_min = m[0] if m[0] < m[1] else m[1]    #next best mu_look
                        #print("mu_min={0} > {1} ? mu_max={2}".format(mu_min, self.T, mu_max))
                        if mu_min > self.T or mu_min <= self.T: #This condition is not required
                            cdecoded[ii] = 0 if m[0]<m[1] else 1
                            decoded[ii] = encoded[0] if m[0]<m[1] else encoded[1]
                            self.updateBITS(decoded[ii], ii)
                            bmetric[j] = mu_min #mu_cur
                            bmetric_cut[j] = mu_max #if bmetric_cut[j] > mu_max else bmetric_cut[j]
                            pm[i] = mu_min
                            state_cur[j] = self.cur_state 
                            self.cur_state = branchState[cdecoded[ii]]
                            if m[0]==m[1]: #They might get very close but it never happens
                                print("m0==m1?")
                            if self.prnt_proc==1 or (self.prnt_proc==3 and on_ml_path == 0 and self.ml_exploring_mu_min_idx == j):
                                print("v={5}, u={4}: min({2:0.6f}, {3:0.6f}) > T={1}, bm_cut={9:0.2f}, nState({8})={7}, pm={6:0.3f}".format(i,self.T,m[0],m[1],decoded[ii],cdecoded[ii],mu_min,self.cur_state,j,bmetric_cut[j]))
                            #bm_fz_pre = 0
                            smaller_bm_followed[j] = 1
                            i += 1
                            j += 1
                            visited_before = 0
                        else:   #This part is not used because there is code in move_back the does the same thing.
                            #Segment-wise
                            print("******************************************************************************************************************************************")
                            """if j == 0 or (on_ml_path == 1 and (i<=31 or (i<=63 and i>39))):
                            #if j == 0:
                                self.T = self.T - self.Delta
                                visited_before = 0
                                if self.prnt_proc==1:
                                    print("T: {0} => {1}, mu_min(u{2}) < T, Move FWD to the best node".format(self.T + self.Delta,self.T,i))
                            else:
                                if self.prnt_proc==1:
                                    print("u{0}: min({1:0.3f}, {2:0.3f}) < T:{3}, cState({5})={4}, Look BACK".format(i,m[0],m[1],self.T,self.cur_state,j))
                                state_cur[j] = self.cur_state
                                
                                [self.T, j, visited_before, visit_worst_idx] = self.move_back_stem(mu_min, bmetric, bmetric_cut, j, self.T, smaller_bm_followed, decoded, CS_flag, mu_max_stem)
                                self.cur_state = [0 for mm in range(self.m)] if j==0 else state_cur[j]
                                if visited_before == 0 and j==0: 
                                    on_ml_path = 1
                                else:
                                    on_ml_path = 0
                                if self.prnt_proc==1:
                                    print("cState({1})={0}".format(self.cur_state,j))
                                i = self.A[j]
                                #bm_fz_pre = 0
                                #i = pcfun.bitreversed(ii1, self.n)"""

                else:
                    #Segment-wise
                    if j == 0 or (on_ml_path == 1 and ( i<=31 or (i<=63 and i>39) )):#i<=31 or (i<=63 and i>39)  or (i<=95 and i>71)
                    #if j == 0:
                        while mu_max < self.T: #Adjusting the threshold in the begning
                            self.T = self.T - self.Delta
                        visited_before = 0
                        if self.prnt_proc==1:
                            print("$$$$ T reduced to {0}, mu_max(u{1}) < T, Move FWD to the best node".format(self.T,i))
                    else:
                        if self.prnt_proc==1:
                            print("u{0}: max({1:0.3f}, {2:0.3f}) < T:{3}, cState({5})={4}, Look BACK".format(i,m[0],m[1],self.T,self.cur_state,j))
                        state_cur[j] = self.cur_state
                        #Reduce treshold down to mu_max to avoid several cycles
                        """if on_ml_path == 1 and (i==39 or i==71):# 
                            while mu_max < self.T - self.Delta: #Adjusting the threshold in the begning
                                self.T = self.T - self.Delta"""
                        #if on_ml_path == 1:
                            #ml_down_explored_idx = 
                        #updated worst node PM   #bmetric_cut[j] should change as well
                        """for k in range(j):#self.information_size): #reduced clck_avg by introducing error
                            if smaller_bm_followed[k] == 1:
                                #self.ml_exploring_mu_min_idx = k #
                                smaller_bm_followed_on_stem[k] = 1
                                #next_ml_idx_to_explore = i
                                if bmetric_cut_updated[k] < mu_max:
                                     bmetric_cut_updated[k] = mu_max
                                     if self.prnt_proc==1 or self.prnt_proc==3:
                                        print("$$$$$$$ Updating bmetric_cut[{0}] = {1:0.2f}************".format(k,mu_max))
                                break"""
                        #It does the same as the above.
                        #"""
                        if on_ml_path == 0:
                            if bmetric_cut_updated[self.ml_exploring_mu_min_idx] < mu_max:
                                bmetric_cut_updated[self.ml_exploring_mu_min_idx] = mu_max
                                if self.prnt_proc==1 or self.prnt_proc==3:
                                    print("$$$$$$$ Updating bmetric_cut[{0}] = {1:0.2f}************".format(self.ml_exploring_mu_min_idx,mu_max))
                        #"""      
                        #Updating biases based on the noise
                        if bias_updated == 0 and on_ml_path == 1 and i == bit_idx_B_updating: #Extra condition are essential
                            if mu_max < B:
                                alpha0 = np.ceil(np.abs(mu_max/B)/B_delta)*B_delta
                                alpha = alpha0 #if alpha0 > ?
                                if self.prnt_proc==1 or self.prnt_proc==3:
                                    print("&&&&&&&& Updating alpha={0:0.2f}, B={1:0.2f} < mu={2:0.2f}".format(alpha,B,mu_max))
                            bias_updated = 1
                            for j2 in range(j):
                                bmetric_cut[j2] = bmetric_cut[j2] + (alpha-1)*Bc[j2]
                            pm[self.A[0]-1] = pm[self.A[0]-1] + (alpha-1)*Bc0
                            pm[i-1] = pm[i-1] + (alpha-1)*Bc1
                            pm[i] = pm[i-1] + np.log(p_max) - alpha*np.log(1-self.pe[ii]) #mu_max
                            #mu_max = pm[i]
                        if on_ml_path == 1:
                            ml_idx_before_exploring = j
                            is_ml_exploring_mu_min = 1
                            ml_path_end = 1
                            self.ml_last_mu_max = mu_max
                            if self.prnt_proc==1 or self.prnt_proc==3:
                                print("****Examining subtrees of ml_path, last ml bit j={0},i={1}****".format(j,i))
                            #mu_max_stem = mu_max
                            
                        [self.T, j1, visited_before] = self.move_back_stem_topdown(mu_max, bmetric, bmetric_cut, j, self.T, smaller_bm_followed, decoded, CS_flag, ml_path_end)
                        if visited_before == 0 and (j1==self.ml_exploring_mu_min_idx or j1==j): #this part may switch to = 0 wrongly when it proceeds to decoding w/o going to origin
                            on_ml_path = 1
                            #if self.prnt_proc==1 or self.prnt_proc==3:
                                #print("Entering ml_path")
                        else:
                            on_ml_path = 0
                            #if self.prnt_proc==1 or self.prnt_proc==3:
                                #print("Leaving ml_path")
                        j=j1
                        ml_path_end = 0
                        self.cur_state = [0 for mm in range(self.m)] if j==0 else state_cur[j]
                        if self.prnt_proc==1:
                            print("cState({1})={0}".format(self.cur_state,j))
                        i = self.A[j]
                        #bm_fz_pre = 0
                        #i = pcfun.bitreversed(ii1, self.n)
                    
            else:
                #if dLLR < 0:
                #if j>0:
                    #bmetric[0 if j==0 else j-1] += np.log((np.exp(dLLR)/(np.exp(dLLR)+1))/(1-self.pe[ii]))
                ##bm_fz_pre += np.log((np.exp(dLLR)/(np.exp(dLLR)+1))/(1-self.pe[ii]))
                encoded[0] = pcfun.conv_1bit(0, self.cur_state, self.gen)
                branchState[0] = pcfun.getNextState(0, self.cur_state, self.m)
                p_max = np.exp(abs_dLLR)/(np.exp(abs_dLLR)+1)
                p_min = 1/(np.exp(abs_dLLR)+1)
                #if on_ml_path == 1:
                    #p_min = p_min if p_min>0.1 else 0.1
                bm = np.log((p_max if (encoded[0]==0 and dLLR>0) or (encoded[0]==1 and dLLR<0) else p_min))
                #print(bm)
                #if on_ml_path == 1:
                    #bm = bm if bm>-2 else -2
                pm[i] = pm[i-1] + bm -alpha*np.log(1-self.pe[ii])
                #pm[i] = pm[i-1] + np.log(p_max if (encoded[0]==0 and dLLR>0) or (encoded[0]==1 and dLLR<0) else p_min) - 0.7
                self.cur_state = branchState[0]
                state_cur[j] = self.cur_state #j
                if self.prnt_proc==1:
                    print("{3}: dllr={0:0.2f}, Pu0={1:0.2f}, 1-Pe={2:0.6f}".format(dLLR,(np.exp(dLLR)/(np.exp(dLLR)+1)),(1-self.pe[ii]),i))
                """else:
                    bmetric[0 if j==0 else j-1] += np.log(1-(np.exp(dLLR)/(np.exp(dLLR)+1))/(1-self.pe[ii]))"""

                cdecoded[ii] = 0
                decoded[ii] = encoded[0]
                if self.prnt_proc==1:
                    print("v{0}=FZ, u{0}={3}: T={1}, cState({5})={4}, pm={2:0.3f}".format(i,self.T,pm[i],encoded[0],self.cur_state,j))
                self.updateBITS(decoded[ii], ii)
                i += 1
            
        if self.prnt_proc==4:
            self.tot_err_freq[tot_err_corrected] += 1
        return self.extract(cdecoded)







        
    def PACfano_stem_decoder(self, soft_mess, issystematic: bool):
        mu_pre = -1000  #-infinity when mu_cur is the root node
        #mu_cur = 0
        #mu_look= 0
        #from_child_node = 0 #0: previous node was parent node, 1: was child node
        bmetric = [0.0 for i in range(self.information_size)]
        bmetric_cut = [0.0 for i in range(self.information_size)]
        bmetric_cut_updated = [-250.0 for i in range(self.information_size)]
        pm = [0 for i in range(self.codeword_length)]
        #bm_fz_pre = 0
        smaller_bm_followed = [0 for i in range(self.information_size)]
        smaller_bm_followed_on_stem = [0 for i in range(self.information_size)]
        visited_before = 0
        m = [0, 0]
        # reset LLRs
        self.LLRs = [0 for i in range(2 * self.codeword_length - 1)]
        # reset BITS
        self.BITS = [[0 for i in range(self.codeword_length - 1)] for j in range(2)]
        # reset decoding results
        decoded = [0 for i in range(self.codeword_length)]
        cdecoded = [0 for i in range(self.codeword_length)]
        # initial LLRs
        self.LLRs[self.codeword_length - 1:] = soft_mess
        frozen_set = (self.polarcode_mask + 1) % 2  #in bitrevesal order
        critical_set = pcfun.generate_critical_set(frozen_set)
        CS_flag = [0 for i in range(self.information_size)]
        encoded = [0, 0]
        branchState = [[],[]]
        state_cur = [[] for i in range(self.information_size)]
        i = 0
        i_pre = -1
        j = 0
        j_cs = 0
        B = 0
        B_delta = 0.2
        alpha = 1
        on_ml_path = 1
        restart = 0
        tot_err_corrected = 0
        #stack
        seg_border = [39, 71, 127]
        seg_idx = 0
        bit_idx_B_updating = 39
        Bc = [0.0 for i in range(self.information_size)]
        for k in range(self.codeword_length):
            B += alpha*np.log(1-self.pe[k])
        Bcmplt = B
        j1=0
        for k in range(self.codeword_length):
            ii0 = pcfun.bitreversed(k, self.n)
            Bcmplt -= alpha*np.log(1-self.pe[ii0])
            if pcfun.bitreversed(self.A[0] - 1, self.n) == ii0:
                Bc0 = Bcmplt
            if pcfun.bitreversed(bit_idx_B_updating - 1, self.n) == ii0:
                Bc1 = Bcmplt
            if self.rate_profile[k] == 1:
                Bc[j1] = Bcmplt
                j1+=1
        """for k in range(self.codeword_length):
            Bcmplt -= alpha*np.log(1-self.pe[k])
            if self.A[0] - 1 == k:
                Bc0 = Bcmplt"""
        pm[-1]+=B
        if self.prnt_proc==1:
            print("B={0}".format(B))
        mu_max_stem = 0
        visit_worst_idx = 0
        bias_updated = 0
        while i < self.codeword_length:
            if i > seg_border[0] and i <= seg_border[1] and on_ml_path == 1:
                seg_idx = 1
            elif i <= seg_border[0] and on_ml_path == 1:
                seg_idx = 0
            elif i > seg_border[1] and on_ml_path == 1:
                seg_idx = 2
                    
            if self.iter_clocks>254*32 and restart == 0: #No significabt effect on clocks_ave
                self.T = -1000
                j = 0
                i  = 0
                restart = 1
                print("Clocks exceeded the cap")
            if i != i_pre: #to avoid recalculation after reucing T in the begining
                i_pre = i
                ii = pcfun.bitreversed(i, self.n)
                
                self.updateLLR(ii)
                dLLR = self.LLRs[0]
                abs_dLLR = np.abs(dLLR)
                self.total_steps += 1
            #print(ii)
            #if self.total_clocks>self.max_clocks:
                #self.T = -100
            p_y = np.log(0.5/(np.sqrt(2*3.14)*self.sigma)*(np.exp(-np.square(soft_mess[ii]-1)/(2*np.square(self.sigma)))+np.exp(-np.square(soft_mess[ii]+1)/(2*np.square(self.sigma)))))
            #print(soft_mess[ii], self.sigma, p_y)
            if self.polarcode_mask[ii] == 1:
                #jj = 0 if j==0 else j-1
                ##mu_pre = 0 if j == 0 else bmetric[j-1] #mu_pre = bmetric[jj]
                #Calc mu_look:
                encoded[0] = pcfun.conv_1bit(0, self.cur_state, self.gen)
                branchState[0] = pcfun.getNextState(0, self.cur_state, self.m)
                encoded[1] = 1 - encoded[0]
                branchState[1] = pcfun.getNextState(1, self.cur_state, self.m)
                #@3.5dB, p_max: RuntimeWarning: invalid value encountered in double_scalars
                p_max = np.exp(abs_dLLR)/(np.exp(abs_dLLR)+1)
                p_min = 1-p_max #1/(np.exp(abs_dLLR)+1)
                #when you try to evaluate log with 0 (not dividng by zero):
                #RuntimeWarning: divide by zero encountered in log
                b = alpha*np.log(1-self.pe[ii])
                p_min += 10**-7
                if dLLR > 0: #p_u0>p_u1
                    m[0] = pm[i-1] + np.log((p_max if encoded[0]==0 else p_min)) - b
                    m[1] = pm[i-1] + np.log((p_max if encoded[1]==0 else p_min)) - b
                    #m[0] = pm[i-1] + np.log(p_max if encoded[0]==0 else p_min)
                    #m[1] = pm[i-1] + np.log(p_max if encoded[1]==0 else p_min)
                    if self.prnt_proc==1:
                        print("{4}: dllr={0:0.2f}, Pu0={1:0.2f} Pu1={2:0.2f}, 1-Pe={3:0.5f}".format(dLLR,(np.exp(dLLR)/(np.exp(dLLR)+1)),(1/(np.exp(dLLR)+1)),(1-self.pe[ii]),i))
                else:   #p_u1>p_u0
                    m[0] = pm[i-1] + np.log((p_max if encoded[0]==1 else p_min)) - b
                    m[1] = pm[i-1] + np.log((p_max if encoded[1]==1 else p_min)) - b
                    #m[0] = pm[i-1] + np.log(p_max if encoded[0]==1 else p_min)
                    #m[1] = pm[i-1] + np.log(p_max if encoded[1]==1 else p_min)
                    if self.prnt_proc==1:
                        print("{4}: dllr={0:0.2f}, Pu0={2:0.2f} Pu1={1:0.2f}, 1-Pe={3:0.5f}".format(dLLR,(np.exp(dLLR)/(np.exp(dLLR)+1)),(1/(np.exp(dLLR)+1)),(1-self.pe[ii]),i))
                mu_max = m[0] if m[0] > m[1] else m[1]    #best mu_look
                mu_min = m[0] if m[0] < m[1] else m[1]    #next best mu_look
                #the commented part: to expand subtrees segment-wise >> no sig. change in complx
                #introduces a slight error despite slight reduction in cplx
                if mu_max >= self.T: #or (mu_max >= (self.T-self.Delta) and i < seg_border[seg_idx] and on_ml_path != 1):
                    #It could be moving forward after backward movement(s):
                    if visited_before == 0:
                        #Move forward:
                        #From now on, mu_look --> mu_cur
                        cdecoded[ii] = 0 if m[0]>m[1] else 1
                        #decoded[ii] = encoded[0] if m[0]>m[1] else encoded[1]
                        decoded[ii] = encoded[cdecoded[ii]]
                        """if bmetric[j] < bmetric_cut[j] and smaller_bm_followed[j] == 1: #They may have been swapped when exploring worst node..
                            tmp = bmetric[j]
                            bmetric[j] = bmetric_cut[j]
                            bmetric_cut[j] = tmp"""
                        bmetric[j] = mu_max #mu_cur (after forward move), previously mu_look
                        #bmetric_cut[j] = mu_min
                        #updated worst node PM
                        bmetric_cut[j] = bmetric_cut_updated[j] if (on_ml_path == 1 and smaller_bm_followed_on_stem[j] == 1) else mu_min
                        pm[i] = mu_max
                        #bm_fz_pre = 0
                        smaller_bm_followed[j] = 0
                        #Ginie corrects the decoded bits, when T is very small:
                        if self.prnt_proc==4:
                            if self.trdata[ii] != cdecoded[ii]:
                                cdecoded[ii] = 1 - cdecoded[ii]
                                decoded[ii] = encoded[cdecoded[ii]]
                                bmetric[j] = mu_min #mu_cur (after forward move), previously mu_look
                                bmetric_cut[j] = mu_max
                                pm[i] = mu_min
                                smaller_bm_followed[j] = 1
                                #Statistics:
                                self.bit_err_cnt[i] += 1
                                tot_err_corrected += 1
                        self.updateBITS(decoded[ii], ii)
                        #print(j)

                        state_cur[j] = self.cur_state    #Current state
                        self.cur_state = branchState[cdecoded[ii]]  #Next state
                        if self.prnt_proc==1:
                            print("v={5}, u={4}: max({2:0.3f},{3:0.3f}) > T={1}, nState({8})={7}, pm={6:0.3f}, Clks={9}".format(i,self.T,m[0],m[1],decoded[ii],cdecoded[ii],mu_max,self.cur_state,j,self.total_clocks))
                        if critical_set[j_cs] == i and CS_flag[j] != 1:
                            CS_flag[j] = 1
                            if j_cs < critical_set.size - 1:
                                j_cs += 1
                        #mu_pre = 0 if j == 0 else bmetric[j-1]#line453 does the same but more accurate for j=0
                        mu_pre = pm[i-1]
                        #T = mu_max
                        #Tightening the threshold in the first visit: Two conditionsfor increasing T
                        """if mu_pre < T + self.Delta: #i.e. n_pre cannot have been visited with higher threshold: previous visit of n_pre was the first visit: n_cur is also visited for the first time.
                            while bmetric[j] > T + self.Delta: #When T is too low; several cycles
                                T = T + self.Delta"""
                        i += 1
                        j += 1
                    else:
                        #mu_min = m[0] if m[0] < m[1] else m[1]    #next best mu_look
                        if mu_min > self.T:
                            cdecoded[ii] = 0 if m[0]<m[1] else 1
                            decoded[ii] = encoded[0] if m[0]<m[1] else encoded[1]
                            self.updateBITS(decoded[ii], ii)
                            bmetric[j] = mu_min #mu_cur
                            bmetric_cut[j] = mu_max #if bmetric_cut[j] > mu_max else bmetric_cut[j]
                            pm[i] = mu_min
                            state_cur[j] = self.cur_state 
                            self.cur_state = branchState[cdecoded[ii]]
                            if self.prnt_proc==1:
                                print("v={5}, u={4}: min({2:0.3f}, {3:0.3f}) > T={1}, nState({8})={7}, pm={6:0.3f}".format(i,self.T,m[0],m[1],decoded[ii],cdecoded[ii],mu_min,self.cur_state,j))
                            #bm_fz_pre = 0
                            smaller_bm_followed[j] = 1
                            i += 1
                            j += 1
                            visited_before = 0
                        else:   #This part is not used because there is code in move_back the does the same thing.
                            #Segment-wise
                            if j == 0 or (on_ml_path == 1 and (i<=31 or (i<=63 and i>39))):
                            #if j == 0:
                                self.T = self.T - self.Delta
                                visited_before = 0
                                if self.prnt_proc==1:
                                    print("T: {0} => {1}, mu_min(u{2}) < T, Move FWD to the best node".format(self.T + self.Delta,self.T,i))
                            else:
                                if self.prnt_proc==1:
                                    print("u{0}: min({1:0.3f}, {2:0.3f}) < T:{3}, cState({5})={4}, Look BACK".format(i,m[0],m[1],self.T,self.cur_state,j))
                                state_cur[j] = self.cur_state
                                
                                [self.T, j, visited_before, visit_worst_idx] = self.move_back_stem(mu_min, bmetric, bmetric_cut, j, self.T, smaller_bm_followed, decoded, CS_flag, mu_max_stem)
                                self.cur_state = [0 for mm in range(self.m)] if j==0 else state_cur[j]
                                if visited_before == 0 and j==0: 
                                    on_ml_path = 1
                                else:
                                    on_ml_path = 0
                                if self.prnt_proc==1:
                                    print("cState({1})={0}".format(self.cur_state,j))
                                i = self.A[j]
                                #bm_fz_pre = 0
                                #i = pcfun.bitreversed(ii1, self.n)

                else:
                    #Segment-wise
                    if j == 0 or (on_ml_path == 1 and (i<=31 or (i<=63 and i>39) )):#i<=31 or (i<=63 and i>39)
                    #if j == 0:
                        while mu_max < self.T: #Adjusting the threshold in the begning
                            self.T = self.T - self.Delta
                        visited_before = 0
                        if self.prnt_proc==1:
                            print("T reduced to {0}, mu_max(u{1}) < T, Move FWD to the best node".format(self.T,i))
                    else:
                        if self.prnt_proc==1:
                            print("u{0}: max({1:0.3f}, {2:0.3f}) < T:{3}, cState({5})={4}, Look BACK".format(i,m[0],m[1],self.T,self.cur_state,j))
                        state_cur[j] = self.cur_state
                        #Reduce treshold down to mu_max to avoid several cycles
                        """if on_ml_path == 1 and (i==39 or i==71):# 
                            while mu_max < self.T - self.Delta: #Adjusting the threshold in the begning
                                self.T = self.T - self.Delta"""
                                
                        #updated worst node PM   #bmetric_cut[j] should change as well
                        for k in range(self.information_size): #reduced clck_avg by introducing error
                            if smaller_bm_followed[k] == 1:
                                if bmetric_cut_updated[k] < mu_max:
                                    smaller_bm_followed_on_stem[k] = 1
                                    bmetric_cut_updated[k] = mu_max
                                    if self.prnt_proc==1 or self.prnt_proc==3:
                                        print("***updated bmetric_cut[{0}] = {1}************".format(k,mu_max))
                                break
                                
                        #Updating biases based on the noise
                        if bias_updated == 0 and on_ml_path == 1 and i==bit_idx_B_updating: #Extra condition are essential
                            if mu_max < B:
                                alpha0 = np.floor(np.abs(mu_max/B)/B_delta)*B_delta
                                alpha = alpha0 #if alpha0 > ?
                                if self.prnt_proc==1 or self.prnt_proc==3:
                                    print("********Updated alpha={0:0.2f}, B={1:0.2f} < mu={2:0.2f}".format(alpha,B,mu_max))
                            bias_updated = 1
                            for j2 in range(j):
                                bmetric_cut[j2] = bmetric_cut[j2] + (alpha-1)*Bc[j2]
                            pm[self.A[0]-1] = pm[self.A[0]-1] + (alpha-1)*Bc0
                            pm[i-1] = pm[i-1] + (alpha-1)*Bc1
                           
                        if on_ml_path == 1:
                            mu_max_stem = mu_max
                            
                        [self.T, j, visited_before, visit_worst_idx] = self.move_back_stem(mu_max, bmetric, bmetric_cut, j, self.T, smaller_bm_followed, decoded, CS_flag, mu_max_stem)
                        if visited_before == 0 and j==0: #this part may switch to = 0 wrongly when it proceeds to decoding w/o going to origin
                            on_ml_path = 1
                        else:
                            on_ml_path = 0
                        self.cur_state = [0 for mm in range(self.m)] if j==0 else state_cur[j]
                        if self.prnt_proc==1:
                            print("cState({1})={0}".format(self.cur_state,j))
                        i = self.A[j]
                        #bm_fz_pre = 0
                        #i = pcfun.bitreversed(ii1, self.n)
                    
            else:
                #if dLLR < 0:
                #if j>0:
                    #bmetric[0 if j==0 else j-1] += np.log((np.exp(dLLR)/(np.exp(dLLR)+1))/(1-self.pe[ii]))
                ##bm_fz_pre += np.log((np.exp(dLLR)/(np.exp(dLLR)+1))/(1-self.pe[ii]))
                encoded[0] = pcfun.conv_1bit(0, self.cur_state, self.gen)
                branchState[0] = pcfun.getNextState(0, self.cur_state, self.m)
                p_max = np.exp(abs_dLLR)/(np.exp(abs_dLLR)+1)
                p_min = 1/(np.exp(abs_dLLR)+1)
                #if on_ml_path == 1:
                    #p_min = p_min if p_min>0.1 else 0.1
                bm = np.log((p_max if (encoded[0]==0 and dLLR>0) or (encoded[0]==1 and dLLR<0) else p_min))
                #print(bm)
                #if on_ml_path == 1:
                    #bm = bm if bm>-2 else -2
                pm[i] = pm[i-1] + bm -alpha*np.log(1-self.pe[ii])
                #pm[i] = pm[i-1] + np.log(p_max if (encoded[0]==0 and dLLR>0) or (encoded[0]==1 and dLLR<0) else p_min) - 0.7
                self.cur_state = branchState[0]
                state_cur[j] = self.cur_state #j
                if self.prnt_proc==1:
                    print("{3}: dllr={0:0.2f}, Pu0={1:0.2f}, 1-Pe={2:0.6f}".format(dLLR,(np.exp(dLLR)/(np.exp(dLLR)+1)),(1-self.pe[ii]),i))
                """else:
                    bmetric[0 if j==0 else j-1] += np.log(1-(np.exp(dLLR)/(np.exp(dLLR)+1))/(1-self.pe[ii]))"""

                cdecoded[ii] = 0
                decoded[ii] = encoded[0]
                if self.prnt_proc==1:
                    print("v{0}=FZ, u{0}={3}: T={1}, cState({5})={4}, pm={2:0.3f}".format(i,self.T,pm[i],encoded[0],self.cur_state,j))
                self.updateBITS(decoded[ii], ii)
                i += 1
            
        if self.prnt_proc==4:
            self.tot_err_freq[tot_err_corrected] += 1
        return self.extract(cdecoded)


    def PACfano_decoder(self, soft_mess, issystematic: bool):
        """SC-decoder
        symbol_energy -  the BPSK symbol energy (linear scale);
        noise_power -  Noise power spectral density (default N0/2 = 1)"""
        #self.Delta = 5 #to tradeoff b/w complexity and FER
        #self.T = -1000   #Threshold
        mu_pre = -1000  #-infinity when mu_cur is the root node
        #mu_cur = 0
        #mu_look= 0
        #from_child_node = 0 #0: previous node was parent node, 1: was child node
        bmetric = [0.0 for i in range(self.information_size)]
        bmetric_cut = [0.0 for i in range(self.information_size)]
        bmetric_cut_updated = [-250.0 for i in range(self.information_size)]
        pm = [0 for i in range(self.codeword_length)]
        #bm_fz_pre = 0
        smaller_bm_followed = [0 for i in range(self.information_size)]
        smaller_bm_followed_on_stem = [0 for i in range(self.information_size)]
        visited_before = 0
        m = [0, 0]
        # reset LLRs
        self.LLRs = [0 for i in range(2 * self.codeword_length - 1)]
        # reset BITS
        self.BITS = [[0 for i in range(self.codeword_length - 1)] for j in range(2)]
        # reset decoding results
        decoded = [0 for i in range(self.codeword_length)]
        cdecoded = [0 for i in range(self.codeword_length)]
        # initial LLRs
        self.LLRs[self.codeword_length - 1:] = soft_mess
        frozen_set = (self.polarcode_mask + 1) % 2  #in bitrevesal order
        critical_set = pcfun.generate_critical_set(frozen_set)
        CS_flag = [0 for i in range(self.information_size)]
        encoded = [0, 0]
        branchState = [[],[]]
        state_cur = [[] for i in range(self.information_size)]
        i = 0
        i_pre = -1
        j = 0
        j_cs = 0
        B = 0
        alpha = 1
        on_ml_path = 1
        restart = 0
        tot_err_corrected = 0
        for k in range(self.codeword_length):
            B += alpha*np.log(1-self.pe[k])
        #pm[-1]+=B
        if self.prnt_proc==1:
            print("B={0}".format(B))
        while i < self.codeword_length:
            """if self.iter_clocks>400000 and restart == 0: #No significabt effect on clocks_ave
                self.T = -1000
                j = 0
                i  = 0
                restart = 1
                print("Clocks exceeded the cap")"""
            if i != i_pre: #to avoid recalculation after reucing T in the begining
                i_pre = i
                ii = pcfun.bitreversed(i, self.n)
                
                self.updateLLR(ii)
                dLLR = self.LLRs[0]
                abs_dLLR = np.abs(dLLR)
                self.total_steps += 1
            #print(ii)
            #if self.total_clocks>self.max_clocks:
                #self.T = -100
            #p_y = np.log(0.5/(np.sqrt(2*3.14)*self.sigma)*(np.exp(-np.square(soft_mess[ii]-1)/(2*np.square(self.sigma)))+np.exp(-np.square(soft_mess[ii]+1)/(2*np.square(self.sigma)))))
            #print(soft_mess[ii], self.sigma, p_y)
            if self.polarcode_mask[ii] == 1:
                #jj = 0 if j==0 else j-1
                ##mu_pre = 0 if j == 0 else bmetric[j-1] #mu_pre = bmetric[jj]
                #Calc mu_look:
                encoded[0] = pcfun.conv_1bit(0, self.cur_state, self.gen)
                branchState[0] = pcfun.getNextState(0, self.cur_state, self.m)
                encoded[1] = 1 - encoded[0]
                branchState[1] = pcfun.getNextState(1, self.cur_state, self.m)
                eLLR = np.exp(abs_dLLR)
                p_max = eLLR /(eLLR +1) #invalid vaalue in double_scalers. #NaN or 
                p_min = 1-p_max #1/(np.exp(abs_dLLR)+1)
                #when you try to evaluate log with 0 (not dividng by zero):
                #RuntimeWarning: divide by zero encountered in log
                p_min += 10**-7
                if dLLR > 0: #p_u0>p_u1
                    m[0] = pm[i-1] + np.log((p_max if encoded[0]==0 else p_min))-alpha*np.log(1-self.pe[ii])
                    m[1] = pm[i-1] + np.log((p_max if encoded[1]==0 else p_min))-alpha*np.log(1-self.pe[ii])
                    #m[0] = pm[i-1] + np.log(p_max if encoded[0]==0 else p_min)
                    #m[1] = pm[i-1] + np.log(p_max if encoded[1]==0 else p_min)
                    if self.prnt_proc==1:
                        print("{4}: dllr={0:0.2f}, Pu0={1:0.2f} Pu1={2:0.2f}, 1-Pe={3:0.5f}".format(dLLR,(np.exp(dLLR)/(np.exp(dLLR)+1)),(1/(np.exp(dLLR)+1)),(1-self.pe[ii]),i))
                    #m[0] = bm_fz_pre + mu_pre + np.log((np.exp(dLLR)/(np.exp(dLLR)+1))/(1-self.pe[ii]))
                    #m[1] = bm_fz_pre + mu_pre + np.log((1 - np.exp(dLLR)/(np.exp(dLLR)+1))/(1-self.pe[ii]))
                else:   #p_u1>p_u0
                    m[0] = pm[i-1] + np.log((p_max if encoded[0]==1 else p_min))-alpha*np.log(1-self.pe[ii])
                    m[1] = pm[i-1] + np.log((p_max if encoded[1]==1 else p_min))-alpha*np.log(1-self.pe[ii])
                    #m[0] = pm[i-1] + np.log(p_max if encoded[0]==1 else p_min)
                    #m[1] = pm[i-1] + np.log(p_max if encoded[1]==1 else p_min)
                    if self.prnt_proc==1:
                        print("{4}: dllr={0:0.2f}, Pu0={2:0.2f} Pu1={1:0.2f}, 1-Pe={3:0.5f}".format(dLLR,(np.exp(dLLR)/(np.exp(dLLR)+1)),(1/(np.exp(dLLR)+1)),(1-self.pe[ii]),i))
                    #m[1] = bm_fz_pre + mu_pre + np.log((1/(np.exp(dLLR)+1))/(1-self.pe[ii]))
                    #m[0] = bm_fz_pre + mu_pre + np.log((1-1/(np.exp(dLLR)+1))/(1-self.pe[ii]))
                mu_max = m[0] if m[0] > m[1] else m[1]    #best mu_look
                mu_min = m[0] if m[0] < m[1] else m[1]    #next best mu_look
                
                if mu_max >= self.T:
                    #It could be moving forward after backward movement(s):
                    if visited_before == 0:
                        #Move forward:
                        #From now on, mu_look --> mu_cur
                        cdecoded[ii] = 0 if m[0]>m[1] else 1
                        #decoded[ii] = encoded[0] if m[0]>m[1] else encoded[1]
                        decoded[ii] = encoded[cdecoded[ii]]
                        """if bmetric[j] < bmetric_cut[j] and smaller_bm_followed[j] == 1: #They may have been swapped when exploring worst node..
                            tmp = bmetric[j]
                            bmetric[j] = bmetric_cut[j]
                            bmetric_cut[j] = tmp"""
                        bmetric[j] = mu_max #mu_cur (after forward move), previously mu_look
                        bmetric_cut[j] = mu_min
#                        updated worst node PM
                        #bmetric_cut[j] = bmetric_cut_updated[j] if (on_ml_path == 1 and smaller_bm_followed_on_stem[j] == 1) else mu_min
                        pm[i] = mu_max
                        #bm_fz_pre = 0
                        smaller_bm_followed[j] = 0
                        #Ginie corrects the decoded bits, when T is very small:
                        if self.prnt_proc==4:
                            if self.trdata[ii] != cdecoded[ii]:
                                cdecoded[ii] = 1 - cdecoded[ii]
                                decoded[ii] = encoded[cdecoded[ii]]
                                bmetric[j] = mu_min #mu_cur (after forward move), previously mu_look
                                bmetric_cut[j] = mu_max
                                pm[i] = mu_min
                                smaller_bm_followed[j] = 1
                                #Statistics:
                                self.bit_err_cnt[i] += 1
                                tot_err_corrected += 1
                        self.updateBITS(decoded[ii], ii)
                        #print(j)

                        state_cur[j] = self.cur_state    #Current state
                        self.cur_state = branchState[cdecoded[ii]]  #Next state
                        if self.prnt_proc==1:
                            print("v={5}, u={4}: max({2:0.3f},{3:0.3f}) > T={1}, nState({8})={7}, pm={6:0.3f}, Clks={9}".format(i,self.T,m[0],m[1],decoded[ii],cdecoded[ii],mu_max,self.cur_state,j,self.total_clocks))
                        if critical_set[j_cs] == i and CS_flag[j] != 1:
                            CS_flag[j] = 1
                            if j_cs < critical_set.size - 1:
                                j_cs += 1
                        #mu_pre = 0 if j == 0 else bmetric[j-1]#line453 does the same but more accurate for j=0
                        mu_pre = pm[i-1]
                        #T = mu_max
                        #Tightening the threshold in the first visit: Two conditionsfor increasing T
                        """if mu_pre < T + self.Delta: #i.e. n_pre cannot have been visited with higher threshold: previous visit of n_pre was the first visit: n_cur is also visited for the first time.
                            while bmetric[j] > T + self.Delta: #When T is too low; several cycles
                                T = T + self.Delta"""
                        i += 1
                        j += 1
                    else:
                        #mu_min = m[0] if m[0] < m[1] else m[1]    #next best mu_look
                        if mu_min > self.T:
                            cdecoded[ii] = 0 if m[0]<m[1] else 1
                            decoded[ii] = encoded[0] if m[0]<m[1] else encoded[1]
                            self.updateBITS(decoded[ii], ii)
                            bmetric[j] = mu_min #mu_cur
                            bmetric_cut[j] = mu_max #if bmetric_cut[j] > mu_max else bmetric_cut[j]
                            pm[i] = mu_min
                            state_cur[j] = self.cur_state 
                            self.cur_state = branchState[cdecoded[ii]]
                            if self.prnt_proc==1:
                                print("v={5}, u={4}: min({2:0.3f}, {3:0.3f}) > T={1}, nState({8})={7}, pm={6:0.3f}".format(i,self.T,m[0],m[1],decoded[ii],cdecoded[ii],mu_min,self.cur_state,j))
                            #bm_fz_pre = 0
                            smaller_bm_followed[j] = 1
                            i += 1
                            j += 1
                            visited_before = 0
                        else:   #This part is not used because there is code in move_back the does the same thing.
                            #Segment-wise
                            if j == 0: #or (on_ml_path == 1 and (i<=31 or (i<=63 and i>39))):
                            #if j == 0:
                                self.T = self.T - self.Delta
                                visited_before = 0
                                if self.prnt_proc==1:
                                    print("T: {0} => {1}, mu_min(u{2}) < T, Move FWD to the best node".format(self.T + self.Delta,self.T,i))
                            else:
                                if self.prnt_proc==1:
                                    print("u{0}: min({1:0.3f}, {2:0.3f}) < T:{3}, cState({5})={4}, Look BACK".format(i,m[0],m[1],self.T,self.cur_state,j))
                                state_cur[j] = self.cur_state
                                
                                [self.T, j, visited_before] = self.move_back(mu_min, bmetric, bmetric_cut, j, self.T, smaller_bm_followed, decoded, CS_flag)
                                self.cur_state = [0 for mm in range(self.m)] if j==0 else state_cur[j]
                                if visited_before == 0 and j==0:
                                    on_ml_path = 1
                                else:
                                    on_ml_path = 0
                                if self.prnt_proc==1:
                                    print("cState({1})={0}".format(self.cur_state,j))
                                i = self.A[j]
                                #bm_fz_pre = 0
                                #i = pcfun.bitreversed(ii1, self.n)

                else:
                    #Segment-wise
                    if j == 0: #or (on_ml_path == 1 and (i<=31 or (i<=63 and i>39) )):#i<=31 or (i<=63 and i>39)
                    #if j == 0:
                        while mu_max < self.T: #Adjusting the threshold in the begning
                            self.T = self.T - self.Delta
                        visited_before = 0
                        if self.prnt_proc==1:
                            print("T reduced to {0}, mu_max(u{1}) < T, Move FWD to the best node".format(self.T,i))
                    else:
                        if self.prnt_proc==1:
                            print("u{0}: max({1:0.3f}, {2:0.3f}) < T:{3}, cState({5})={4}, Look BACK".format(i,m[0],m[1],self.T,self.cur_state,j))
                        state_cur[j] = self.cur_state
                        """if on_ml_path == 1 and (i==39 or i==71):# 
                            while mu_max < self.T - self.Delta: #Adjusting the threshold in the begning
                                self.T = self.T - self.Delta"""
                        #updated worst node PM
                        """for k in range(self.information_size): #reduced clck_avg by introducing error
                            if smaller_bm_followed[k] == 1:
                                if bmetric_cut_updated[k] < mu_max:
                                    smaller_bm_followed_on_stem[k] = 1
                                    bmetric_cut_updated[k] = mu_max
                                    if self.prnt_proc==1 or self.prnt_proc==3:
                                        print("***updated bmetric_cut[{0}] = {1}************".format(k,pm_cur))
                                break"""
                        [self.T, j, visited_before] = self.move_back(mu_max, bmetric, bmetric_cut, j, self.T, smaller_bm_followed, decoded, CS_flag)
                        if visited_before == 0 and j==0:
                            on_ml_path = 1
                        else:
                            on_ml_path = 0
                        self.cur_state = [0 for mm in range(self.m)] if j==0 else state_cur[j]
                        if self.prnt_proc==1:
                            print("cState({1})={0}".format(self.cur_state,j))
                        i = self.A[j]
                        #bm_fz_pre = 0
                        #i = pcfun.bitreversed(ii1, self.n)
                    
            else:
                #if dLLR < 0:
                #if j>0:
                    #bmetric[0 if j==0 else j-1] += np.log((np.exp(dLLR)/(np.exp(dLLR)+1))/(1-self.pe[ii]))
                ##bm_fz_pre += np.log((np.exp(dLLR)/(np.exp(dLLR)+1))/(1-self.pe[ii]))
                encoded[0] = pcfun.conv_1bit(0, self.cur_state, self.gen)
                branchState[0] = pcfun.getNextState(0, self.cur_state, self.m)
                p_max = np.exp(abs_dLLR)/(np.exp(abs_dLLR)+1)
                p_min = 1/(np.exp(abs_dLLR)+1)
                #if on_ml_path == 1:
                    #p_min = p_min if p_min>0.1 else 0.1
                bm = np.log((p_max if (encoded[0]==0 and dLLR>0) or (encoded[0]==1 and dLLR<0) else p_min))
                #print(bm)
                #if on_ml_path == 1:
                    #bm = bm if bm>-2 else -2
                pm[i] = pm[i-1] + bm -alpha*np.log(1-self.pe[ii])
                #pm[i] = pm[i-1] + np.log(p_max if (encoded[0]==0 and dLLR>0) or (encoded[0]==1 and dLLR<0) else p_min) - 0.7
                self.cur_state = branchState[0]
                state_cur[j] = self.cur_state #j
                if self.prnt_proc==1:
                    print("{3}: dllr={0:0.2f}, Pu0={1:0.2f}, 1-Pe={2:0.6f}".format(dLLR,(np.exp(dLLR)/(np.exp(dLLR)+1)),(1-self.pe[ii]),i))
                """else:
                    bmetric[0 if j==0 else j-1] += np.log(1-(np.exp(dLLR)/(np.exp(dLLR)+1))/(1-self.pe[ii]))"""

                cdecoded[ii] = 0
                decoded[ii] = encoded[0]
                if self.prnt_proc==1:
                    print("v{0}=FZ, u{0}={3}: T={1}, cState({5})={4}, pm={2:0.3f}".format(i,self.T,pm[i],encoded[0],self.cur_state,j))
                self.updateBITS(decoded[ii], ii)
                i += 1
            
        if self.prnt_proc==4:
            self.tot_err_freq[tot_err_corrected] += 1
        return self.extract(cdecoded)




    def scfano_decode(self, soft_mess, issystematic: bool):
        """SC-decoder
        symbol_energy -  the BPSK symbol energy (linear scale);
        noise_power -  Noise power spectral density (default N0/2 = 1)"""
        #self.Delta = 1 #to tradeoff b/w complexity and FER
        T = self.T
        #T = -1000#-1   #Threshold
        mu_pre = -1000  #-infinity when mu_cur is the root node
        #mu_cur = 0
        #mu_look= 0
        #from_child_node = 0 #0: previous node was parent node, 1: was child node
        bmetric = [0 for i in range(self.information_size)]
        bmetric_cut = [0 for i in range(self.information_size)]
        pm = [0 for i in range(self.codeword_length)]
        #bm_fz_pre = 0
        smaller_bm_followed = [0 for i in range(self.information_size)]
        visited_before = 0
        m = [0, 0]
        # reset LLRs
        self.LLRs = [0 for i in range(2 * self.codeword_length - 1)]
        # reset BITS
        self.BITS = [[0 for i in range(self.codeword_length - 1)] for j in range(2)]
        # reset decoding results
        decoded = [0 for i in range(self.codeword_length)]
        # initial LLRs
        self.LLRs[self.codeword_length - 1:] = soft_mess
        frozen_set = (self.polarcode_mask + 1) % 2  #in bitrevesal order
        critical_set = pcfun.generate_critical_set(frozen_set)
        CS_flag = [0 for i in range(self.information_size)]
        i = 0
        j = 0
        j_cs = 0
        while i < self.codeword_length:
            ii = pcfun.bitreversed(i, self.n)
            self.updateLLR(ii)
            dLLR = self.LLRs[0]
            #print(ii)
            if self.polarcode_mask[ii] == 1:
                #jj = 0 if j==0 else j-1
                ##mu_pre = 0 if j == 0 else bmetric[j-1] #mu_pre = bmetric[jj]
                #Calc mu_look:
                if dLLR > 0:
                    m[0] = pm[i-1] + np.log((np.exp(dLLR)/(np.exp(dLLR)+1))/(1.0-self.pe[ii]))
                    m[1] = pm[i-1] + np.log((1/(np.exp(dLLR)+1))/(1.0-self.pe[ii]))
                    if self.prnt_proc==1:
                        print("{4}: dllr={0:0.2f}, Pu0={1:0.2f} Pu1={2:0.2f}, 1-Pe={3:0.5f}".format(dLLR,(np.exp(dLLR)/(np.exp(dLLR)+1)),(1/(np.exp(dLLR)+1)),(1-self.pe[ii]),i))
                    #m[0] = bm_fz_pre + mu_pre + np.log((np.exp(dLLR)/(np.exp(dLLR)+1))/(1-self.pe[ii]))
                    #m[1] = bm_fz_pre + mu_pre + np.log((1 - np.exp(dLLR)/(np.exp(dLLR)+1))/(1-self.pe[ii]))
                else:
                    m[1] = pm[i-1] + np.log((1/(np.exp(dLLR)+1))/(1.0-self.pe[ii]))
                    m[0] = pm[i-1] + np.log((np.exp(dLLR)/(np.exp(dLLR)+1))/(1.0-self.pe[ii]))
                    if self.prnt_proc==1:
                        print("{4}: dllr={0:0.2f}, Pu0={2:0.2f} Pu1={1:0.2f}, 1-Pe={3:0.5f}".format(dLLR,(np.exp(dLLR)/(np.exp(dLLR)+1)),(1/(np.exp(dLLR)+1)),(1-self.pe[ii]),i))
                    #m[1] = bm_fz_pre + mu_pre + np.log((1/(np.exp(dLLR)+1))/(1-self.pe[ii]))
                    #m[0] = bm_fz_pre + mu_pre + np.log((1-1/(np.exp(dLLR)+1))/(1-self.pe[ii]))
                mu_max = m[0] if m[0]>m[1] else m[1]    #best mu_look
                mu_min = m[0] if m[0] < m[1] else m[1]    #next best mu_look
                
                if mu_max >= T:
                    #It could be moving forward after backward movement(s):
                    if visited_before == 0:
                        #Move forward:
                        #From now on, mu_look --> mu_cur
                        decoded[ii] = 0 if m[0]>m[1] else 1
                        if self.prnt_proc==1:
                            print("u{0}={4}: max({2:0.3f}, {3:0.3f}) > T={1}, pm={5:0.3f}".format(i,T,m[0],m[1],decoded[ii],mu_max))
                        self.updateBITS(decoded[ii], ii)
                        #print(j)
                        bmetric[j] = mu_max #mu_cur (after forward move), previously mu_look
                        bmetric_cut[j] = mu_min
                        if critical_set[j_cs] == i and CS_flag[j_cs] != 1:
                            CS_flag[j_cs] = 1
                            if j_cs < critical_set.size - 1:
                                j_cs += 1
                        pm[i] = mu_max
                        #bm_fz_pre = 0
                        smaller_bm_followed[j] = 0
                        #mu_pre = 0 if j == 0 else bmetric[j-1]#line453 does the same but more accurate for j=0
                        mu_pre = pm[i-1]
                        #Tightening the threshold in the first visit: Two conditionsfor increasing T
                        if mu_pre < T + self.Delta: #i.e. n_pre cannot have been visited with higher threshold: previous visit of n_pre was the first visit: n_cur is also visited for the first time.
                            while bmetric[j] > T + self.Delta: #When T is too low; several cycles
                                T = T + self.Delta
                        i += 1
                        j += 1
                    else:
                        #mu_min = m[0] if m[0] < m[1] else m[1]    #next best mu_look
                        if mu_min > T:
                            decoded[ii] = 0 if m[0]<m[1] else 1
                            if self.prnt_proc==1:
                                print("u{0}={4}: min({2:0.3f}, {3:0.3f}) > T={1}, pm={5:0.3f}".format(i,T,m[0],m[1],decoded[ii],mu_min))
                            self.updateBITS(decoded[ii], ii)
                            bmetric[j] = mu_min #mu_cur
                            bmetric_cut[j] = mu_max
                            pm[i] = mu_min
                            #bm_fz_pre = 0
                            smaller_bm_followed[j] = 1
                            i += 1
                            j += 1
                            visited_before = 0
                        else:
                            if j == 0:
                                T = T - self.Delta
                                visited_before = 0
                                if self.prnt_proc==1:
                                    print("T: {0} => {1}, mu_min(u{2}) < T, Move FWD to the best node".format(T + self.Delta,T,i))
                            else:
                                if self.prnt_proc==1:
                                    print("u{0}: min({1:0.3f}, {2:0.3f}) < T:{3}, Look BACK".format(i,m[0],m[1],T))
                                [T, j, visited_before] = self.move_back(mu_max, bmetric, bmetric_cut, j, T, smaller_bm_followed, decoded, CS_flag)
                                i = self.A[j]
                                #bm_fz_pre = 0
                                #i = pcfun.bitreversed(ii1, self.n)

                else:
                    if j == 0:
                        while mu_max < T: #Adjusting the threshold in the begning
                            T = T - self.Delta
                        visited_before = 0
                        if self.prnt_proc==1:
                            print("T reduced to {0}, mu_max(u{1}) < T, Move FWD to the best node".format(T,i))
                    else:
                        if self.prnt_proc==1:
                            print("u{0}: max({1:0.3f}, {2:0.3f}) < T:{3}, Look BACK".format(i,m[0],m[1],T))
                        [T, j, visited_before] = self.move_back(mu_max, bmetric, bmetric_cut, j, T, smaller_bm_followed, decoded, CS_flag)
                        i = self.A[j]
                        #bm_fz_pre = 0
                        #i = pcfun.bitreversed(ii1, self.n)
                    
            else:
                #if dLLR < 0:
                #if j>0:
                    #bmetric[0 if j==0 else j-1] += np.log((np.exp(dLLR)/(np.exp(dLLR)+1))/(1-self.pe[ii]))
                ##bm_fz_pre += np.log((np.exp(dLLR)/(np.exp(dLLR)+1))/(1-self.pe[ii]))
                pm[i] = pm[i-1] + np.log((np.exp(dLLR)/(np.exp(dLLR)+1))/(1-self.pe[ii]))
                if self.prnt_proc==1:
                    print("{3}: dllr={0:0.2f}, Pu0={1:0.2f}, 1-Pe={2:0.6f}".format(dLLR,(np.exp(dLLR)/(np.exp(dLLR)+1)),(1-self.pe[ii]),i))
                """else:
                    bmetric[0 if j==0 else j-1] += np.log(1-(np.exp(dLLR)/(np.exp(dLLR)+1))/(1-self.pe[ii]))"""

                decoded[ii] = 0
                if self.prnt_proc==1:
                    print("u{0}:Frozen: T={1}, pm={2:0.3f}".format(i,T,pm[i]))
                self.updateBITS(decoded[ii], ii)
                i += 1
            

        if issystematic:
            self.mul_matrix(decoded)
            return self.extract(decoded)
        return self.extract(decoded)




