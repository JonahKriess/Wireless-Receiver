import numpy as np

# Helper to load complex numbers from a text file where i is used instead of j
def load_complex_file(filename):
    with open(filename, 'r') as f:
        data = f.read().replace('i', 'j')
    return np.array([complex(x) for x in data.split()])


# Downconvert the signal to baseband by mixing with a complex exponential at the carrier frequency
def downconvert(rx, center_freq=20, sample_freq=100):
    n = np.arange(len(rx))
    t = n/sample_freq
    carrier = np.exp(-1j*2*np.pi*center_freq*t)
    baseband = rx * carrier
    return baseband

# Apply a low-pass filter in the frequency domain by zeroing out FFT bins above the cutoff frequency
def lowpass_filter(bb, sample_freq=100, cutoff=5.1):
    N = len(bb)
    fft_sig = np.fft.fft(bb, n=N)
    freqs = np.fft.fftfreq(N, d=1/sample_freq)
    mask = np.abs(freqs) <= cutoff
    filtered = fft_sig * mask
    filtered_time_domain = np.fft.ifft(filtered)
    return filtered_time_domain

# Downsample the signal by taking every nth sample
def downsample(sig, factor=10):
    return sig[::factor]

# Find the start of the message symbols by correlating with the known preamble 
def find_start(downsampled, preamble):
    corr = np.correlate(downsampled, preamble.conj(), mode='valid')
    start_idx = np.argmax(np.abs(corr))
    return start_idx, corr

# 16-QAM demodulation by mapping the I and Q components to the nearest constellation points
def demodulate_16QAM(symbols):
    levels = np.array([-3, -1, 1, 3])
    detected_symbols = []

    for symbol in symbols:
        I_hat = levels[np.argmin(np.abs(levels - np.real(symbol)))]
        Q_hat = levels[np.argmin(np.abs(levels - np.imag(symbol)))]

        detected_symbols.append(complex(I_hat, Q_hat))
    return detected_symbols

# Map the detected symbols to bits using I_MAP and Q_MAP
def map_detected_to_bits(detected):
    bits = ""
    for s in detected:
        I = int(np.real(s))
        Q = int(np.imag(s))
        bits += Q_MAP[Q] + I_MAP[I]
    return bits

# Convert the bitstring to ASCII message, stopping at a NUL byte and trimming to full bytes
def bitstring_to_ascii(bitstring):
    message = ""
    for i in range(0, len(bitstring) - 7, 8):
        byte = int(bitstring[i:i+8], 2)
        if byte == 0:
            break
        message += chr(byte)
    return message

if __name__ == "__main__":

    # Load files for received signal and preamble
    rx = np.loadtxt("input.txt")
    preamble = load_complex_file("preamble.txt")

    I_MAP = {
        -3: "10",
        -1: "11",
         1: "01",
         3: "00",
    }

    Q_MAP = {
        -3: "00",
        -1: "01",
         1: "11",
         3: "10",
    }

    # 1. Downconvert
    baseband = downconvert(rx)

    # 2. Low-pass filter in frequency domain
    filtered = lowpass_filter(baseband)

    # 3. Downsample (symbol rate = 10 Hz)
    ds_rx = downsample(filtered, factor=10)

    # 4. Find start of message symbols
    start_idx, _ = find_start(ds_rx, preamble)

    # 5. Extract symbols after preamble
    symbols = ds_rx[start_idx + len(preamble):]

    # 6. Normalize average power to ideal 16QAM power (10)
    avg_power = np.mean(np.abs(symbols)**2)
    symbols = symbols / np.sqrt(avg_power / 10)

    # 7. Hard decision 16-QAM demodulation
    detected = demodulate_16QAM(symbols)

    # 8. Map detected symbols to bitstring using I_MAP and Q_MAP
    bitstring = map_detected_to_bits(detected)

    # 9. Convert bitstring to ASCII message
    message = bitstring_to_ascii(bitstring)

    print("Decoded message:")
    print(message)