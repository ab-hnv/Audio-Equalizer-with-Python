import numpy as np
import scipy.io.wavfile as wavfile

# FFT function
def fft(x, inverse=False):
    N = len(x)
    if N <= 1:
        return x
    even = fft(x[0::2], inverse=inverse)
    odd = fft(x[1::2], inverse=inverse)
    factor = 2j * np.pi / N if inverse else -2j * np.pi / N
    T = [np.exp(factor * k) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

try:
    # Read the audio signal from a WAV file
    sampling_rate, audio_signal = wavfile.read('sample-6s.wav')

    # Define the equalization parameters
    frequency_bands = [(0, 500), (500, 2000), (2000, sampling_rate // 2)]
    gain_values = [1.6, 100, 3.0]

    # Split the audio signal into frames with overlap
    frame_size = 1024
    overlap = 0.5
    overlap_size = int(frame_size * overlap)
    step_size = frame_size - overlap_size

    # Apply audio equalization
    output_signal = np.zeros(len(audio_signal), dtype=np.float64)
    for i in range(0, len(audio_signal) - frame_size, step_size):
        # Extract the current frame
        frame = audio_signal[i:i + frame_size].astype(np.float64)
        frame = audio_signal[i:i + frame_size].mean(axis=1).astype(np.float64)
        # Apply window function to the frame
        window = np.hamming(frame_size)
        windowed_frame = frame * window

        # Compute the FFT of the windowed frame
        fft_frame = fft(windowed_frame)

        # Modify the frequency components based on the gain values
        for j, (start_freq, end_freq) in enumerate(frequency_bands):
            start_bin = int(start_freq * frame_size / sampling_rate)
            end_bin = int(end_freq * frame_size / sampling_rate)

            fft_frame[start_bin:end_bin] = np.array(fft_frame[start_bin:end_bin]) * gain_values[j]


        # Compute the IFFT of the modified frame
        ifft_frame = fft(fft_frame, inverse=True)

        # Overlap and add the frames to reconstruct the output signal
        output_signal[i:i + frame_size] += np.real(ifft_frame)

    # Normalize the output signal to the original range
    output_signal = np.int16(output_signal / np.max(np.abs(output_signal)) * 32767)

    # Write the equalized audio signal to a WAV file
    wavfile.write('output_audio.wav', sampling_rate, output_signal)

    print("Audio equalization completed successfully.")

except FileNotFoundError:
    print("Input audio file not found.")
except Exception as e:
    print("An error occurred during audio equalization:", str(e))
