from clean_artifacts import clean_artifacts

def clean_rawdata(EEG, arg_flatline, arg_highpass, arg_channel, arg_noisy, arg_burst, arg_window):
    print('The function clean_rawdata has been deprecated and is only kept for backward')
    print('compatibility. Use the clean_artifacts function instead.')

    if arg_flatline == -1:
        arg_flatline = 'off'
        print('flatchan rej disabled.')

    if arg_highpass == -1:
        arg_highpass = 'off'
        print('highpass disabled.')

    if arg_channel == -1:
        arg_channel = 'off'
        print('badchan rej disabled.')

    if arg_noisy == -1:
        arg_noisy = 'off'
        print('noise-based rej disabled.')

    if arg_burst == -1:
        arg_burst = 'off'
        print('burst clean disabled.')

    if arg_window == -1:
        arg_window = 'off'
        print('bad window rej disabled.')

    cleanEEG = clean_artifacts(EEG, FlatlineCriterion=arg_flatline,
                                    Highpass=arg_highpass,
                                    ChannelCriterion=arg_channel,
                                    LineNoiseCriterion=arg_noisy,
                                    BurstCriterion=arg_burst,
                                    WindowCriterion=arg_window)

    return cleanEEG