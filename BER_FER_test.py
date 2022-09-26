from polar_code import PolarCode
from time import time
import polar_coding_functions as pcf
import csv
import numpy as np

#from multiprocessing import Process, Queue
#import multiprocessing
#import threading
import numpy
from channel import channel

class BERFER():
    """structure that keeps results of BER and FER tests"""
    def __init__(self):
        self.fname = str()
        self.label = str()
        self.snr_range = list()
        self.ber = list()
        self.fer = list()
        self.fer_init = list()
        self.clocks = list()
        self.steps = list()
        self.cmplx = list()
        self.operations = list()
        self.mod_type = str()
        self.xrange = list()


def fails(list1, list2):
    """returns number of bit errors"""
    return pcf.np.sum(pcf.np.absolute(list1 - list2))


def ber_fer_pac_test(code, snr_range, conv_gen, mem, systematic=False, sclist=False, tests=10000): # -> BERFER : for documentation
    result = BERFER()
    result.xrange = snr_range
    
    code.m = mem
    code.gen = conv_gen
    code.cur_state = [0 for i in range(mem)]
    log_M = 1   #M:modulation order
    T_init = code.T
    

    for snr in snr_range:
        code.pe = pcf.pe_dega(code.codeword_length,
                                                   code.information_size,
                                                   code.designSNR)
        print("\nSNR={0} dB".format(snr))
        result.snr_range.append(snr)
        Rc = (code.information_size / code.codeword_length)
        #Ec = N0 * Rc * pow(10, snr / 10)
        #snr = snr + 10*numpy.log10(Rc)
        if code.snrb_snr == 'SNR':
            N0 = 1 / pow(10, snr / 10)  #Noise power
        else:
            N0 = 1 / (Rc*pow(10, snr / 10))  #Noise power
        sigma = numpy.sqrt(N0/2)
        #sigma = numpy.sqrt(N0/(2*Rc))
       ##sigma = numpy.sqrt(1/(2*Rc))*numpy.power(10,-snr/20) #for Eb/N0
        code.sigma = sigma
        #sigma_s = numpy.sqrt(0.5)*numpy.power(10,-snr/20)
        #sigma = sigma_b
        ber = 0
        fer = 0
        ch = channel(code.modu, N0)
        for t in range(tests):
            code.iter_clocks = 0
            code.iter_clocks = 0
            code.T = T_init
            code.cur_state = [0 for i in range(mem)] #This line wasn't here initially and that was the cause of degradation
            message = pcf.gen_messages(code.information_size)
            x = code.pac_encode(message, conv_gen, mem)
            #x = code.encode(u, issystematic=systematic)
            modulated = ch.modulate(x)
            y = ch.add_noise(modulated,N0)
            llr_ch = ch.calc_llr(y)
            decoded = code.PACfano_decoder(llr_ch, issystematic=systematic)
            #decoded = code.PACfano_stem_decoder(llr_ch, issystematic=systematic)
            #decoded = code.PACfano_stem_topdown_decoder(llr_ch, issystematic=systematic)
            code.total_clocks += code.iter_clocks

            ber += fails(message, decoded)
            if not pcf.np.array_equal(message, decoded):
                for idx in range(code.information_size):
                    if message[idx] !=decoded[idx]:
                        #err_idx = pcf.bitreversed(code.A[idx], code.n)
                        err_idx = code.A[idx]
                        break
                fer += 1
                print("Error {0} t={1}, errIndx={6} => FER={2:0.2e}, Clks={3}, Clks_avg={4:0.0f}, operations={5:0.0f}".format(fer,t, fer/(t+1), code.iter_clocks, code.total_clocks/(t+1), (code.total_additions+code.total_comparisons) / (t + 1), err_idx))
            #fer += not pcf.np.array_equal(message, decoded)
            if fer > 199 and tests > 1000:    #//:Floor Division
                print("@ {0} dB FER is {1:0.2e}".format(snr, fer/(t+1)))
                break
            if t%2000==0:
                print("t={0} FER={1} ".format(t, fer/(t+1)))
        #print("{0} ".format(ber))
        result.ber.append(ber / ((t + 1) * code.information_size))
        result.fer.append(fer / (t + 1))
        result.clocks.append(code.total_clocks / (t + 1))
        result.operations.append((code.total_additions+code.total_comparisons) / (t + 1))
        result.steps.append(code.total_steps / (t + 1))
        result.cmplx.append(code.total_clocks/ (2*code.codeword_length-2) / (t + 1))
        code.total_clocks = 0
        code.total_steps = 0
        code.total_additions = 0
        code.total_comparisons = 0
    # filename for saving results
    if True:
        result.fname = "abc"
    #if type(code) == ShortenPolarCode:
    #    result.fname = code.shorten_method
    result.fname += "P({0},{1})".format(code.codeword_length, code.information_size)
    """
    if systematic:
        result.fname += "_sys"
    else:
        result.fname += "_nonsys"
    if sclist:
        result.fname += "_SCLD({0})".format(code.list_size)
    else:
        result.fname += "_sc"
        """
    # writing resuls in file
    with open(result.fname + ".csv", 'w') as f:
        """if systematic and sclist:
            result.label = "Systematic ({0}, {1}) polar code;\nwith design SNR = {2};\n" \
                     "SC List decoding with L = {3}\n".format(code.codeword_length, code.information_size,
                                                             code.designSNR, code.list_size)
        elif systematic and not sclist:
            result.label = "Systematic ({0}, {1}) polar code;\nwith design SNR = {2};\n" \
                     "SC decoding\n".format(code.codeword_length, code.information_size, code.designSNR)
        elif not systematic and sclist:
            result.label = "Non-systematic ({0}, {1}) polar code;\nwith design SNR = {2};\n" \
                     "SC List decoding with L = {3}\n".format(code.codeword_length, code.information_size,
                                                             code.designSNR, code.list_size)
        else:
            result.label = "Non-systematic ({0}, {1}) polar code;\nwith design SNR = {2};\n" \
                     "SC decoding\n".format(code.codeword_length, code.information_size, code.designSNR)"""
        result.label="def"
        f.write(result.label)

        for snr in result.snr_range:
            f.write("{0}; ".format(snr))
        f.write("\n")
        for ber in result.ber:
            f.write("{0}; ".format(ber))
        f.write("\n")
        for fer in result.fer:
            f.write("{0}; ".format(fer))
        for steps in result.steps:
            f.write("{0}; ".format(steps))
        for clocks in result.clocks:
            f.write("{0}; ".format(clocks))
        for cmplx in result.cmplx:
            f.write("{0}; ".format(cmplx))

    print(result.label)
    print("SNR\t{0}".format(result.snr_range))
    print("BER\t{0}".format(result.ber))
    #print("FER\t{0:1.2e}".format(result.fer))
    print("FER\t{0}".format(result.fer))
    print("Clocks\t{0}".format(result.clocks))
    print("Steps\t{0}".format(result.steps))
    print("Operations\t{0}".format(result.operations))
    print("nCmplx\t{0}".format(result.cmplx))
    print("Tot. Error Freq\t{0}".format(code.tot_err_freq))
    if code.prnt_proc==4: #For Gnie-aided 
        with open("err_cnt.csv", 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter = ',', lineterminator = '\n') #Default:'\r\n' used in Unix # creating a csv writer object
            #csvwriter.writerow(row)
            csvwriter.writerows(map(lambda x: [x], code.bit_err_cnt))
    """with open("elim_freq.csv", 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter = ',', lineterminator = '\n') #Default:'\r\n' used in Unix # creating a csv writer object
        #csvwriter.writerow(row)
        csvwriter.writerows(map(lambda x: [x], code.elim_freq))
    with open("corr_pos.csv", 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter = ',', lineterminator = '\n') #Default:'\r\n' used in Unix # creating a csv writer object
        #csvwriter.writerow(row)
        csvwriter.writerows(map(lambda x: [x], code.corr_pos))
    with open("pmr.csv", 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter = ',', lineterminator = '\n') #Default:'\r\n' used in Unix # creating a csv writer object
        #csvwriter.writerow(row)
        csvwriter.writerows(map(lambda x: [x/(t+1)], code.pmr_accum))"""

    return result