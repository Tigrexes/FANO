from time import time
#import plotly
#import plotly.graph_objs as go
import BER_FER_test as pctest
#from tests import BER_FER_test as pctest
import polar_coding_functions as pcf
from polar_code import PolarCode
from rate_profile import rateprofile
import numpy as np

tests = 10**6

N = 2**6
R = 0.5
K = int(N*R)
dsnr = 4
construct = "rm-polar" #"dega"

conv_gen = [1,0,1,1,0,1,1]

m = len(conv_gen)-1

rprofile = rateprofile(N,K, dsnr)

code101 = PolarCode(N, K, construct, dsnr, rprofile)

code101.Delta = 1
code101.T = 0
code101.flips_const = 10 #4
code101.bit_idx_B_updating = 39

code101.modu = 'BPSK'
code101.snrb_snr = 'SNRb' # 'SNRb' 'SNR'
code101.prnt_proc = 0
SNRrange = np.arange(1,5.5,0.5)#arange(start,endpoint+step,step )
print("PAC({},{}) constructed by {}({}dB and gen={})".format(N, K,construct,dsnr,conv_gen))

#Xrange = []
print("BER & FER test is started")

st = time()
results = list()

results.append(pctest.ber_fer_pac_test(code101, SNRrange, conv_gen, m, systematic=False, sclist=True, tests=tests))
print("time on test = ", str(time() - st), '\n------------\n')
#Xrange = set().union(SNRrange, Xrange)  #Union of two iterable and make it a set

# scatter plot for FERs
"""
data_fer = list()
for res in results:
    data_fer.append(go.Scatter(x=res.xrange, y=res.fer, name=res.fname))
layout = go.Layout(
    title="P({0},{1}),SCLD({2})\ndesignSNR = {3}".format(N, K, list_size, dsnr),
    xaxis=dict(
        range=[SNRrange[0], SNRrange[-1]], #list(SNRrange),   #Error?
        showline=True,
        dtick=0.5,
        title="Signal-to-Noise Ratio"
    ),
    yaxis=dict(
        type='log',
        autorange=True,
        showline=True,
        title="Frame Error Rate"
    )
)
fig_fer = go.Figure(data=data_fer, layout=layout)
plotly.offline.plot(fig_fer, filename="FER_{0}_{1}".format(N, K))
"""

"""
# scatter plot for BERs


data_ber = list()
for res in results:
    data_ber.append(go.Scatter(x=res.xrange, y=res.ber, name=res.fname))
layout = go.Layout(
    title="({0}, {1}) Polar code\nDesign SNR = {2}".format(N, K, dsnr),
    xaxis=dict(
        range=list(Xrange),
        showline=True,
        dtick=0.5,
        title="Signal-to-Noise Ratio"
    ),
    yaxis=dict(
        type='log',
        autorange=True,
        showline=True,
        title="Bit Error Rate"
    )
)
fig_ber = go.Figure(data=data_ber, layout=layout)
plotly.offline.plot(fig_ber, filename="BER_{0}_{1}".format(N, K))
"""





