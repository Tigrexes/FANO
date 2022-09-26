import numpy as np

class channel:
    def __init__(self, modulation, N0):
        self.modulation = modulation
        self.noise_power = N0
        self.code_word_length = 0
        self.sigma = np.sqrt(N0/2)

    def modulate(self, m):
        modulated = []
        self.code_word_length = len(m)
        if self.modulation == 'BPSK':
            modulated = [(1 - 2 * x) for x in m]
        elif self.modulation == 'QPSK':
            if (np.mod(len(m),2)):
                m = [0] + list(m)
            for msb, lsb in zip(m[0::2], m[1::2]):
                modulated.append(-1.0/np.sqrt(2) * (1+1j) + 2.0/np.sqrt(2) * (msb+(lsb)*1j))
        return modulated

    def calc_llr(self,c):
        llr = []
        if self.modulation == 'BPSK':
            llr = [(4/self.noise_power*y) for y in c]
        elif self.modulation=='QPSK':
            for y in c:
                llr += [(-4/np.sqrt(2)*y.real/self.noise_power),(-4/np.sqrt(2)*y.imag/self.noise_power)]
        return np.array(llr)

    def add_noise(self, signal, noise_power):
        if self.modulation == 'BPSK':
            #return signal + noise_power*np.random.randn(len(signal))
            return signal + np.sqrt(noise_power/2)*np.random.standard_normal(len(signal))
        elif self.modulation == 'QPSK':
            return signal + noise_power/np.sqrt(2) * np.random.randn(len(signal)) + noise_power/np.sqrt(2) * np.random.randn(len(signal)) * 1j
