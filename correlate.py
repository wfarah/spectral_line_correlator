import numpy as np
import matplotlib.pyplot as plt

spec_res = 64e6 #MHz

# Main assumption: CPU is providing data with only 1 (big) frequency channel
NANTS  = 20
NFREQS = 1 
NTIME  = 262144
NFREQ_FFT = NTIME
NPOLS  = 2

def correlate_ant_pols(ant1, ant2):
    """
    ant1 and ant2 are 2D arrays with
    shapes [nfreq, npol]

    **********
    *  NOTE: *
    **********
    I am inverse conjugating ant2, an operation that is dublicated
    many times throughout the for loop. A possible 
    """
    corr_xx = ant1[:,0] * np.conj(ant2[:,0])
    corr_xy = ant1[:,0] * np.conj(ant2[:,1])
    corr_yx = ant1[:,1] * np.conj(ant2[:,0])
    corr_yy = ant1[:,1] * np.conj(ant2[:,1])

    # This is another thing we need to check:
    # How are the polarizations structured in the
    # visibilities?
    return np.column_stack((corr_xx, corr_xy, corr_yx, corr_yy))


# block is 8bit+8bit complex
block = np.random.normal(size=(NANTS, NTIME, NPOLS)) +\
        1j*np.random.normal(size=(NANTS, NTIME, NPOLS))


# cast to float 32 (or 16)
block_32 = block.astype(np.complex64)


# FFT block
block_fft = np.zeros_like(block_32)

# Run FFT first
for iant in range(NANTS):
    for ipol in range(NPOLS):
        # take all time samples from single antenna and 
        t_array = block_32[iant, :, ipol]
        
        # now FFT, shift, and store data
        block_fft[iant, :, ipol] = np.fft.fftshift(np.fft.fft(t_array))
        # but also, how about a PFB? :)


# Now for the correlation
nblines = int((NANTS*(NANTS+1))/2)

# I envision a user input for "zoom" mode
# where we only correlate a subset of the frequency channels
# That will help with reduce datasize downstream
zoom_freqs   = [0, NFREQ_FFT]
zoom_n_freqs = zoom_freqs[1] - zoom_freqs[0]

block_corr = np.zeros(shape=(nblines, zoom_n_freqs, NPOLS*NPOLS),
                      dtype = np.complex64)

# Now to correlate

# We index the correlation block with ibline
ibline = 1

# Doing auto-correlations first
for iant1 in range(NANTS):
    ant = block_fft[iant, zoom_freqs[0]:zoom_freqs[1]]
    block_corr[ibline, :] = correlate_ant_pols(ant, ant)
    ibline += 1

# Now cross-correlations
# This is filling in the upper triangular correlation matrix 
# without the diagonal
for iant1 in range(NANTS):
    for iant2 in range(iant1+1, NANTS):
        ant1 = block_fft[iant1, zoom_freqs[0]:zoom_freqs[1]] 
        ant2 = block_fft[iant2, zoom_freqs[0]:zoom_freqs[1]]

        block_corr[ibline, :] = correlate_ant_pols(ant1, ant2)

        ibline += 1


# Pass block_corr to an integrator
# and sum 1-to-1
