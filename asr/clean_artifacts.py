import numpy as np
from .clean_drifts import clean_drifts_iir 
from .clean_flatlines import clean_flatlines
from .asr_utils import pop_select
from .clean_channels import clean_channels
from .clean_channels_nolocs import clean_channels_nolocs
from .clean_asr import clean_asr, clean_asr_realtime
from .clean_windows import clean_windows
import time
import scipy.io

def clean_artifacts(EEG, head_shaking, windowlen, stepsize, maxdims, **kwargs):
    """
    All-in-one function for artifact removal, including ASR.
    
    Parameters:
        EEG: Raw continuous EEG recording to clean up.
        kwargs: Additional optional parameters (specified below).
    
    Optional Parameters:
        channel_criterion: Minimum channel correlation. Default is 0.85.
        line_noise_criterion: Line noise relative criterion. Default is 4.
        burst_criterion: Standard deviation cutoff for removal of bursts. Default is 5.
        window_criterion: Criterion for removing unrepaired time windows. Default is 0.25.
        highpass: Transition band for initial high-pass filter. Default is [0.25, 0.75].
        # ... (other parameters as described in the original MATLAB function)
    
    Returns:
        EEG: Final cleaned EEG recording.
        HP: Optionally just the high-pass filtered data.
        BUR: Optionally the data without final removal of "irrecoverable" windows.
    """
    EEG = EEG.copy()
    head_shaking_patterns = head_shaking.copy()
    
    # Extract and set parameters from keyword arguments (kwargs)
    channel_criterion = kwargs.get('ChannelCorrelationCriterion', 0.8)
    line_crit = kwargs.get('LineNoiseCriterion', 4)
    burst_crit = kwargs.get('BurstCriterion', 5)
    channels = kwargs.get('Channels', None)
    channels_ignore = kwargs.get('Channels_ignore', None)
    window_crit = kwargs.get('WindowCriterion', 0.25)
    num_samples = kwargs.get('NumSamples', 50)
    # highpass_band = kwargs.get('Highpass', [0.25, 0.75])
    highpass_band = kwargs.get('Highpass', 'off')
    channel_crit_maxbad_time = kwargs.get('ChannelCriterionMaxBadTime', 0.5)
    burst_crit_refmaxbadchns = kwargs.get('BurstCriterionRefMaxBadChns', 0.075)
    burst_crit_reftolerances = kwargs.get('BurstCriterionRefTolerances', [-np.inf, 5.5])
    distance2 = kwargs.get('Distance', 'euclidian')
    window_crit_tolerances = kwargs.get('WindowCriterionTolerances', [-3.5, 7])
    burst_rejection = kwargs.get('BurstRejection', 'off')
    nolocs_channel_crit = kwargs.get('NoLocsChannelCriterion', 0.45)
    nolocs_channel_crit_excluded = kwargs.get('NoLocsChannelCriterionExcluded', 0.1)
    max_mem = kwargs.get('MaxMem', 64)
    availableRAM_GB = kwargs.get('availableRAM_GB', None)
    flatline_crit = kwargs.get('FlatlineCriterion', 5)
        
    # ignore some channels
    if channels is not None and channels_ignore is not None:
        raise ValueError("Can include or ignore channel but not both at the same time")
       
    oriEEG = EEG.copy()
    
    if channels is not None:
        EEG = pop_select(EEG, channels)
        EEG['event'] = []  # Clear the events; will be added back later
        
    if channels_ignore is not None:
        EEG = pop_select(EEG, exclude=channels_ignore)
        oriEEG_without_ignored_channels = EEG.copy()
        EEG['event'] = []  # Clear the events; will be added back later

    # remove flat-line channels
    if flatline_crit != 'off':
        # print('Detecting flat line...')
        EEG = clean_flatlines(EEG, flatline_crit)

    # high-pass filter the data
    if highpass_band != 'off':
        print('Applying highpass filter...')
        EEG = clean_drifts_iir(EEG, highpass_band)
        EEG['data'] = EEG['data'].T
    
    oldSrate = EEG['srate']
    EEG['srate'] = round(EEG['srate'])
    removed_channels = []
    
    if channel_criterion != 'off' or line_crit != 'off':
        if channel_criterion == 'off':
            channel_criterion = 0
        if line_crit == 'off':
            line_crit = 100
        
    # EEG, removed_channels = clean_channels_nolocs(EEG, nolocs_channel_crit, nolocs_channel_crit_excluded, 2, channel_crit_maxbad_time, False)
            
    # repair bursts by ASR
    if burst_crit != 'off':
        if distance2.lower() == 'euclidian':    
            # BUR = clean_asr(EEG, cutoff=burst_crit, ref_maxbadchannels=burst_crit_refmaxbadchns, ref_tolerances=burst_crit_reftolerances, maxmem=max_mem)
            # We feed head shaking patterns extracted from IMU data instead of letting the ASR learn itself
            BUR, AF = clean_asr(EEG, head_shaking_patterns, windowlen=windowlen, stepsize=stepsize, maxdims=maxdims, cutoff=burst_crit, ref_maxbadchannels=burst_crit_refmaxbadchns, ref_tolerances=burst_crit_reftolerances, maxmem=max_mem)

        if burst_rejection == 'on':
            # portion of data which have changed
            sample_mask = np.sum(np.abs(EEG['data'] - BUR['data']), axis=0) < 1e-8
            
            # find latency of regions
            retain_data_intervals = np.reshape(np.where(np.diff([False, *sample_mask, False]))[0], (2, -1)).T
            retain_data_intervals[:, 1] -= 1

            # remove small intervals
            if retain_data_intervals.size > 0:
                smallIntervals = np.diff(retain_data_intervals, axis=1).flatten() < 5
                for iInterval in np.where(smallIntervals)[0]:
                    sample_mask[retain_data_intervals[iInterval, 0]:retain_data_intervals[iInterval, 1]] = False
                retain_data_intervals = retain_data_intervals[~smallIntervals, :]

            # reject regions
            EEG = pop_select(EEG, 'point', retain_data_intervals)
            EEG['etc']['clean_sample_mask'] = sample_mask
        else:
            EEG = BUR

    EEG['srate'] = oldSrate

    # remove irrecoverable time windows based on power
    # if window_crit != 'off' and window_crit_tolerances != 'off':
    #     print('Now doing final post-cleanup of the output.')
    #     EEG, sample_mask = clean_windows(EEG, window_crit, window_crit_tolerances)

    print('Use vis_artifacts to compare the cleaned data to the original.')

    return EEG, AF


##########################################################################################
##########################################################################################
###################### Real time function dedicated for GPU ##############################
##########################################################################################
##########################################################################################
# from numba import jit, cuda

# @jit(target_backend='cuda', nopython=True)  
def clean_artifacts_realtime(EEG, **kwargs):
    """
    All-in-one function for artifact removal, including ASR.
    
    Parameters:
        EEG: Raw continuous EEG recording to clean up.
        kwargs: Additional optional parameters (specified below).
    
    Optional Parameters:
        channel_criterion: Minimum channel correlation. Default is 0.85.
        line_noise_criterion: Line noise relative criterion. Default is 4.
        burst_criterion: Standard deviation cutoff for removal of bursts. Default is 5.
        window_criterion: Criterion for removing unrepaired time windows. Default is 0.25.
        highpass: Transition band for initial high-pass filter. Default is [0.25, 0.75].
        # ... (other parameters as described in the original MATLAB function)
    
    Returns:
        EEG: Final cleaned EEG recording.
        HP: Optionally just the high-pass filtered data.
        BUR: Optionally the data without final removal of "irrecoverable" windows.
    """
    EEG = EEG.copy()
    
    # Extract and set parameters from keyword arguments (kwargs)
    channel_criterion = kwargs.get('ChannelCorrelationCriterion', 0.8)
    line_crit = kwargs.get('LineNoiseCriterion', 4)
    burst_crit = kwargs.get('BurstCriterion', 5)
    channels = kwargs.get('Channels', None)
    channels_ignore = kwargs.get('Channels_ignore', None)
    window_crit = kwargs.get('WindowCriterion', 0.25)
    num_samples = kwargs.get('NumSamples', 50)
    # highpass_band = kwargs.get('Highpass', [0.25, 0.75])
    highpass_band = kwargs.get('Highpass', 'off')
    channel_crit_maxbad_time = kwargs.get('ChannelCriterionMaxBadTime', 0.5)
    burst_crit_refmaxbadchns = kwargs.get('BurstCriterionRefMaxBadChns', 0.075)
    burst_crit_reftolerances = kwargs.get('BurstCriterionRefTolerances', [-np.inf, 5.5])
    distance2 = kwargs.get('Distance', 'euclidian')
    window_crit_tolerances = kwargs.get('WindowCriterionTolerances', [-np.inf, 7])
    burst_rejection = kwargs.get('BurstRejection', 'off')
    nolocs_channel_crit = kwargs.get('NoLocsChannelCriterion', 0.45)
    nolocs_channel_crit_excluded = kwargs.get('NoLocsChannelCriterionExcluded', 0.1)
    max_mem = kwargs.get('MaxMem', 64)
    availableRAM_GB = kwargs.get('availableRAM_GB', None)
    flatline_crit = kwargs.get('FlatlineCriterion', 5)
        
    # ignore some channels
    if channels is not None and channels_ignore is not None:
        raise ValueError("Can include or ignore channel but not both at the same time")
    
    EEG['srate'] = round(EEG['srate'])
    
    clean_channel_mask = np.ones(EEG['data'].shape[0], dtype=bool)

    if channel_criterion != 'off' or line_crit != 'off':
        if channel_criterion == 'off':
            channel_criterion = 0
        if line_crit == 'off':
            line_crit = 100
                    
    start_time = time.time()
    # repair bursts by ASR
    if burst_crit != 'off':
        if distance2.lower() == 'euclidian':    
            results = clean_asr_realtime(EEG['data'], EEG['srate'], EEG['nbchan'], clean_channel_mask, cutoff=burst_crit, ref_maxbadchannels=burst_crit_refmaxbadchns, ref_tolerances=np.array(burst_crit_reftolerances), maxmem=max_mem)
    elapsed_time = (time.time() - start_time) * 1000  # multiply by 1000 to convert to milliseconds
    # print("Execution time: ", elapsed_time)
    return results


