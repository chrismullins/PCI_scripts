import mne
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import csv

audvis_dig = mne.channels.read_dig_montage(fif='/home/chris/projects/EEGViewer/Data/sample_audvis_raw.fif')
# dig_montage_new = mne.channels.DigMontage(hsp=tms_montage.pos/1000, point_names=audvis_dig.point_names, hpi=audvis_dig.hpi, elp=audvis_dig.elp, dig_ch_pos=dict(zip([dig_channel_names_dict[ch_name] for ch_name in tms_montage.ch_names], tms_montage.pos.astype('float32')/1000)), lpa=tms_montage.pos[tms_montage.ch_names.index('LPA')]/1000, rpa=tms_montage.pos[tms_montage.ch_names.index('RPA')]/1000, nasion=tms_montage.pos[tms_montage.ch_names.index('Nz')]/1000, dev_head_t=np.eye(4))

tms_montage = mne.channels.read_montage(kind='standard_1005')
# Here is a dict mapping our channel names to 'EEG001' etc...
dig_channel_names_dict = dict(zip(tms_montage.ch_names, ['EEG'+'%03d'%index for index in xrange(len(tms_montage.ch_names))]))
dig_montage_new = mne.channels.DigMontage(hsp=tms_montage.pos/1000, \
                                          point_names=audvis_dig.point_names, \
                                          hpi=audvis_dig.hpi, \
                                          elp=audvis_dig.elp, \
                                          dig_ch_pos=dict(zip([dig_channel_names_dict[ch_name] for ch_name in tms_montage.ch_names], tms_montage.pos.astype('float32')/1000)), \
                                          lpa=tms_montage.pos[tms_montage.ch_names.index('LPA')]/1000, \
                                          rpa=tms_montage.pos[tms_montage.ch_names.index('RPA')]/1000, \
                                          nasion=tms_montage.pos[tms_montage.ch_names.index('Nz')]/1000, \
                                          dev_head_t=np.eye(4))
dig_channel_names_dict = dict(zip(tms_montage.ch_names, ['EEG'+'%03d'%index for index in xrange(len(tms_montage.ch_names))]))

datadir = '/home/chris/projects/EEGViewer/Data/'
vhdr = 'TMS_session_wake.vhdr'
raw_bv = mne.io.read_raw_brainvision(op.join(datadir, vhdr), preload=True)
events_bv = raw_bv.get_brainvision_events()

new_ch_names = []
for ch_name in raw_bv.ch_names:
    if ch_name in dig_channel_names_dict.keys():
        new_ch_names.append(dig_channel_names_dict[ch_name])
    else:
        new_ch_names.append(ch_name)
raw_bv.info['ch_names'] = new_ch_names
# if these don't match up you get a runtime warning
for item in raw_bv.info['chs']:
    if item['ch_name'] in dig_channel_names_dict.keys():
        item['ch_name'] = dig_channel_names_dict[item['ch_name']]
# delete channels we don't have
for key in dig_montage_new.dig_ch_pos.keys():
    if not key in raw_bv.info['ch_names']:
        del dig_montage_new.dig_ch_pos[key]

raw_bv.set_montage(dig_montage_new)


raw_bv.filter(1, 45)
events_trig_on = events_bv[::2]
event_id = dict(stim_on=1)
tmin = -0.5
tmax = 0.5
baseline = (None, 0) # means from the first instant to t = 0
epochs = mne.Epochs(raw_bv, events_trig_on, event_id, tmin, tmax, proj=True, baseline=baseline, preload=True)
epochs.resample(362.5)
evoked = epochs.average()

from mne.datasets import sample
mne_data_path = sample.data_path()

sample_source_space = mne.read_source_spaces(fname='/home/chris/projects/nme-python/mne-venv2/lib/python2.7/site-packages/examples/MNE-sample-data/subjects/sample/bem/sample-oct-6-orig-src.fif')
#sample_bem_surfaces = mne.read_bem_surfaces(fname='/home/chris/projects/nme-python/mne-venv2/lib/python2.7/site-packages/examples/MNE-sample-data/subjects/sample/bem/sample-5120-5120-5120-bem-sol.fif')
fwd_model = mne.make_forward_solution(info=evoked.info, src=sample_source_space, bem='/home/chris/projects/nme-python/mne-venv2/lib/python2.7/site-packages/examples/MNE-sample-data/subjects/sample/bem/sample-5120-5120-5120-bem-sol.fif', eeg=True, meg=False, trans='/home/chris/projects/EEGViewer/Data/sample-trans.fif')
# fwd_model = mne.make_forward_solution(info=evoked.info, src=source_space, bem='/home/chris/projects/nme-python/mne-venv2/lib/python2.7/site-packages/examples/MNE-sample-data/subjects/sample/bem/sample-5120-5120-5120-bem-sol.fif', eeg=True, meg=False, trans='/home/chris/projects/EEGViewer/Data/sample-trans.fif')
cov = mne.compute_covariance(epochs, tmax=0)
inv = mne.minimum_norm.make_inverse_operator(evoked.info, fwd_model, cov, loose=0.2)
stc = mne.minimum_norm.apply_inverse(evoked=evoked, inverse_operator=inv, lambda2=1./9., method='MNE')
