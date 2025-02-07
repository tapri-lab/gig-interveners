"""BEat Consistency metric using Empirical Mode Decomposition (BECEMD)""" 
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter
import emd
import librosa
import matplotlib.pyplot as plt
from bvh import Bvh


# Your existing filter and envelope functions
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filtfilt(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low')
    return b, a

def butter_lowpass_filtfilt(data, cutoff, fs, order=2):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def amp_envelope(audiofilename):
    audio, sr = librosa.load(audiofilename, sr=None)
    data = butter_bandpass_filtfilt(audio, 400, 4000, sr, order=2)
    data = butter_lowpass_filtfilt(np.abs(data), 10, sr, order=2)
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data, sr

def my_get_next_imf(x, zoom=None, sd_thresh=0.1):
    proto_imf = x.copy()
    continue_sift = True
    niters = 0

    if zoom is None:
        zoom = (0, x.shape[0])

    while continue_sift:
        niters += 1
        upper_env = emd.sift.interp_envelope(proto_imf, mode='upper')
        lower_env = emd.sift.interp_envelope(proto_imf, mode='lower')
        avg_env = (upper_env+lower_env) / 2
        stop, val = emd.sift.stop_imf_sd(proto_imf-avg_env, proto_imf, sd=sd_thresh)
        proto_imf = proto_imf - avg_env
        if stop:
            continue_sift = False

    return proto_imf

def detect_motion_beats(signal, threshold=0.2, min_distance=30, fps=200):
    """
    Detect motion beats using find_peaks with adjusted parameters
    """
    peaks, _ = find_peaks(signal, 
                         height=threshold,
                         distance=min_distance,  # Minimum samples between peaks
                         prominence=threshold)   # Minimum prominence of peaks
    
    return peaks / fps

def beat_consistency_score(motion_beats, signal, sr=200, sigma=0.1):
    """
    Compute Beat Consistency Score between motion and audio beats
    """
    if len(motion_beats) == 0:
        return 0.0, []
    
    # Find peaks in signal
    peaks, _ = find_peaks(signal, 
                         height=0.2,
                         distance=30,
                         prominence=0.2)
    
    signal_beats = peaks / sr
    
    if len(signal_beats) == 0:
        return 0.0, []
    
    total_score = 0
    for audio_time in signal_beats:
        min_dist = np.min((audio_time - motion_beats) ** 2)
        score = np.exp(-min_dist / (2 * sigma * sigma))
        total_score += score
    
    return total_score / len(signal_beats), signal_beats

def calculate_angle_changes(angles_data, change_angle_weights=None):
    angle_pairs = [
        ('LeftArm', 'LeftForeArm'),
        ('RightArm', 'RightForeArm'),
        ('LeftForeArm', 'LeftHand'),
        ('RightForeArm', 'RightHand')
    ]
    
    if change_angle_weights is None:
        change_angle_weights = [1.0] * len(angle_pairs)
    
    n_frames = len(next(iter(angles_data.values())))
    angle_diff = np.zeros(n_frames)
    
    for idx, (joint1, joint2) in enumerate(angle_pairs):
        # Get vectors and apply Savgol filter to smooth positions
        vec1 = savgol_filter(angles_data[joint1], window_length=31, polyorder=3, axis=0)
        vec2 = savgol_filter(angles_data[joint2], window_length=31, polyorder=3, axis=0)
        
        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1, axis=1)[:, np.newaxis]
        vec2_norm = vec2 / np.linalg.norm(vec2, axis=1)[:, np.newaxis]
        
        # Calculate angles
        inner_product = np.sum(vec1_norm * vec2_norm, axis=1)
        inner_product = np.clip(inner_product, -1, 1)
        angles = np.arccos(inner_product) / np.pi
        
        # Calculate derivatives and apply Savgol filter again
        angle_changes = np.abs(np.diff(angles))
        angle_changes = np.concatenate(([0], angle_changes))
        angle_changes = savgol_filter(angle_changes, window_length=15, polyorder=3)
        
        angle_diff += angle_changes / change_angle_weights[idx] / len(change_angle_weights)
    
    return angle_diff

def extract_arm_angles(filename):
    with open(filename) as f:
        mocap = Bvh(f.read())
    
    arm_joints = [
        'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
        'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand'
    ]
    
    angles = {}
    for joint in arm_joints:
        if joint in mocap.get_joints_names():
            channels = mocap.joint_channels(joint)
            data = []
            for frame in range(mocap.nframes):
                frame_data = []
                for channel in channels:
                    if 'rotation' in channel:
                        frame_data.append(mocap.frame_joint_channel(frame, joint, channel))
                data.append(frame_data)
            angles[joint] = np.array(data)
    
    frame_time = mocap.frame_time
    print(f"Loaded {len(angles)} joints, {len(next(iter(angles.values())))} frames")
    print(f"Frame time: {frame_time:.3f}s")
    
    return angles, frame_time

def scale_signal(signal):
    """
    Scale signal to range [-1, 1] around zero mean
    """
    # Remove mean
    signal = signal - np.mean(signal)
    # Scale to [-1, 1]
    if np.max(np.abs(signal)) > 0:  # Avoid division by zero
        signal = signal / np.max(np.abs(signal))
    return signal

def compute_becemd(bvh_file, audio_file, plot=False):
    """
    Compute Beat Consistency scores using Empirical Mode Decomposition (BECEMD)
    
    This function analyzes the temporal alignment between motion and audio beats using
    a multi-scale approach through Empirical Mode Decomposition (EMD).
    
    Parameters:
    -----------
    bvh_file : str
        Path to the BVH motion file
    audio_file : str
        Path to the audio WAV file
    plot : bool, optional
        Whether to generate visualization plots (default: False)
        
    Methodology:
    -----------
    1. Motion Beat Detection:
        - Extracts joint angles from BVH file
        - Smooths positions using Savitzky-Golay filter (window=31, order=3)
        - Calculates angle changes and smooths derivatives (window=15, order=3)
        - Detects beats using peak detection with:
            * threshold = 0.1 (normalized signal)
            * minimum distance = 30 samples (150ms at 200Hz)
            * prominence = threshold
    
    2. Audio Beat Detection:
        - Extracts amplitude envelope using bandpass (400-4000Hz) and lowpass (10Hz) filters
        - Resamples to match motion sampling rate (200Hz)
        - Detects beats using same peak detection parameters
    
    3. EMD Processing:
        - Decomposes both signals into IMFs (Intrinsic Mode Functions)
        - Analyzes beat consistency at multiple time scales
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'raw_score': Beat consistency score for raw signals
        - 'imf1_score': Beat consistency score for first IMF
        - 'imf2_score': Beat consistency score for second IMF
        - 'signals': Dictionary of processed signals and detected beats
    """
    # Load and process motion data
    angles, frame_time = extract_arm_angles(bvh_file)
    angle_diff = calculate_angle_changes(angles)
    angle_diff = scale_signal(angle_diff)
    
    # Detect motion beats
    motion_beats = detect_motion_beats(angle_diff, threshold=0.1, min_distance=30)

    # Process audio
    ampv, sr = amp_envelope(audio_file)
    target_len = len(angle_diff)
    ampv = np.interp(np.linspace(0, len(ampv), target_len), 
                     np.arange(len(ampv)), 
                     ampv)
    ampv = scale_signal(ampv)

    # Calculate IMFs
    imf1_audio = scale_signal(my_get_next_imf(ampv))
    imf2_audio = scale_signal(my_get_next_imf(ampv - imf1_audio))
    imf1_motion = scale_signal(my_get_next_imf(angle_diff))
    imf2_motion = scale_signal(my_get_next_imf(angle_diff - imf1_motion))

    # Calculate beat consistency scores
    bc_raw, audio_beats_raw = beat_consistency_score(motion_beats, ampv)
    bc_imf1, audio_beats_imf1 = beat_consistency_score(motion_beats, imf1_audio)
    bc_imf2, audio_beats_imf2 = beat_consistency_score(motion_beats, imf2_audio)

    if plot:
        plt.figure(figsize=(15, 15))

        # Plot raw signals
        plt.subplot(311)
        time_axis = np.arange(len(angle_diff)) / 200
        plt.plot(time_axis, angle_diff, label='Motion', alpha=0.6)
        plt.plot(time_axis, ampv, label='Audio', alpha=0.6)
        plt.vlines(motion_beats, -1, 1, colors='r', label='Motion Beats', alpha=0.5)
        plt.vlines(audio_beats_raw, -1, 1, colors='g', label='Audio Beats', alpha=0.5)
        plt.title(f'Raw Signals (BC Score: {bc_raw:.4f})')
        plt.ylim(-1.1, 1.1)
        plt.legend()

        # Plot IMF1
        plt.subplot(312)
        plt.plot(time_axis, imf1_motion, label='Motion IMF1', alpha=0.6)
        plt.plot(time_axis, imf1_audio, label='Audio IMF1', alpha=0.6)
        plt.vlines(motion_beats, -1, 1, colors='r', label='Motion Beats', alpha=0.5)
        plt.vlines(audio_beats_imf1, -1, 1, colors='g', label='Audio Beats', alpha=0.5)
        plt.title(f'IMF1 (BC Score: {bc_imf1:.4f})')
        plt.ylim(-1.1, 1.1)
        plt.legend()

        # Plot IMF2
        plt.subplot(313)
        plt.plot(time_axis, imf2_motion, label='Motion IMF2', alpha=0.6)
        plt.plot(time_axis, imf2_audio, label='Audio IMF2', alpha=0.6)
        plt.vlines(motion_beats, -1, 1, colors='r', label='Motion Beats', alpha=0.5)
        plt.vlines(audio_beats_imf2, -1, 1, colors='g', label='Audio Beats', alpha=0.5)
        plt.title(f'IMF2 (BC Score: {bc_imf2:.4f})')
        plt.ylim(-1.1, 1.1)
        plt.legend()

        plt.tight_layout()
        plt.show()

    # Return results
    results = {
        'raw_score': bc_raw,
        'imf1_score': bc_imf1,
        'imf2_score': bc_imf2,
        'signals': {
            'motion': angle_diff,
            'audio': ampv,
            'motion_imf1': imf1_motion,
            'audio_imf1': imf1_audio,
            'motion_imf2': imf2_motion,
            'audio_imf2': imf2_audio,
            'motion_beats': motion_beats,
            'audio_beats_raw': audio_beats_raw,
            'audio_beats_imf1': audio_beats_imf1,
            'audio_beats_imf2': audio_beats_imf2
        }
    }
    cleanresults = {'audio_beats_raw': audio_beats_raw,
            'audio_beats_imf1': audio_beats_imf1,
            'audio_beats_imf2': audio_beats_imf2}
    
    return results, cleanresults

# Example usage:
# scores = compute_becemd('../samples/c-cut.bvh', '../samples/audio.wav', plot=True)
# print(f"Raw Score: {scores['raw_score']:.4f}")
# print(f"IMF1 Score: {scores['imf1_score']:.4f}")
# print(f"IMF2 Score: {scores['imf2_score']:.4f}")