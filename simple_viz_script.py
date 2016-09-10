THING_TO_DO = 9

import mne
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import os.path as op
import csv
import mayavi
import sys

datadir = '/home/chris/Projects/MNE-ipython-notebook/MNE-ipython-notebook/Data'
vhdr = 'TMS_session_wake.vhdr'

raw_bv = mne.io.read_raw_brainvision(op.join(datadir, vhdr), preload=True)
events_bv = raw_bv.get_brainvision_events()

if THING_TO_DO == 0:
    raw_bv.plot_psd(fmax=250)
    sys.exit(0)

tms_montage = mne.channels.read_montage(kind='standard_1005')
audvis_dig = mne.channels.read_dig_montage(fif=op.join(datadir,'sample_audvis_raw.fif'))
# Here is a dict mapping our channel names to 'EEG001' etc...
dig_channel_names_dict = dict(zip(tms_montage.ch_names, 
        ['EEG'+'%03d'%index for index in range(len(tms_montage.ch_names))]))
dig_montage_new = mne.channels.DigMontage(
    hsp=tms_montage.pos/1000, \
    point_names=audvis_dig.point_names, \
    hpi=audvis_dig.hpi, \
    elp=audvis_dig.elp, \
    dig_ch_pos=dict(zip([dig_channel_names_dict[ch_name] for ch_name in tms_montage.ch_names], tms_montage.pos.astype('float32')/1000)), \
    lpa=tms_montage.pos[tms_montage.ch_names.index('LPA')]/1000, \
    rpa=tms_montage.pos[tms_montage.ch_names.index('RPA')]/1000, \
    nasion=tms_montage.pos[tms_montage.ch_names.index('Nz')]/1000, \
   dev_head_t=np.eye(4))
dig_channel_names_dict = dict(zip(tms_montage.ch_names, 
    ['EEG'+'%03d'%index for index in range(len(tms_montage.ch_names))]))

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
for key in list(dig_montage_new.dig_ch_pos.keys()):
    if not key in raw_bv.info['ch_names']:
        del dig_montage_new.dig_ch_pos[key]

raw_bv.set_montage(dig_montage_new)
if THING_TO_DO == 1:
    dig_montage_new.plot()
    sys.exit(0)

events_trig_on = events_bv[::2]
event_id = dict(stim_on=1)
tmin = -0.5
tmax = 0.5
baseline = (None, 0) # means from the first instant to t = 0
epochs = mne.Epochs(raw_bv, events_trig_on, event_id, tmin, tmax, proj=True, baseline=baseline, preload=True)
epochs.resample(362.5)
evoked = epochs.average()

evoked.pick_types(meg=False, eeg=True, exclude=['GATE', 'TRIG1', 'TRIG2', 'EOG'])
if THING_TO_DO == 2:
    evoked.plot_topomap(times=np.linspace(-0.2, 0.499, 10), ch_type='eeg')
    sys.exit(0)

subjects_dir = op.join(datadir, 'subjects')
if THING_TO_DO == 3:
    mne.viz.plot_bem(subject='sample', subjects_dir=subjects_dir, orientation='sagittal')
    sys.exit(0)
if THING_TO_DO == 4:
    mne.viz.plot_bem(subject='sample', subjects_dir=subjects_dir, orientation='coronal')
    sys.exit(0)

trans = op.join(subjects_dir, 'sample', 'chris', 'sample-trans.fif')
if THING_TO_DO == 5:
    mne.viz.plot_trans(raw_bv.info, trans, subject='sample', dig=True, subjects_dir=subjects_dir)
    raw_input("Press Enter to continue...")
    sys.exit(0)

surfaces = mne.make_bem_model(subject='sample', subjects_dir=subjects_dir, ico=4)
mne.write_bem_surfaces(fname=op.join(subjects_dir, 'sample', 'chris', 'sample_surface_sol.fif'), surfs=surfaces)
surfaces_read = mne.read_bem_surfaces(op.join(subjects_dir, 'sample', 'chris', 'sample_surface_sol.fif'), patch_stats=True)
if THING_TO_DO == 6:
    head_col = (0.95, 0.83, 0.83)  # light pink
    skull_col = (0.91, 0.89, 0.67)
    brain_col = (0.67, 0.89, 0.91)  # light blue
    colors = [head_col, skull_col, brain_col]
    # 3D source space
    from mayavi import mlab  # noqa

    mlab.figure(size=(600, 600), bgcolor=(0, 0, 0))
    for c, surf in zip(colors, surfaces_read):
        points = surf['rr']
        faces = surf['tris']
        mlab.triangular_mesh(points[:, 0], points[:, 1], points[:, 2], faces,
                                 color=c, opacity=0.3)
        raw_input("Press Enter to plot the next surface (or exit)...")
    sys.exit(0)


src = mne.setup_source_space(subject='sample', spacing='oct6',
                             subjects_dir=subjects_dir,
                             add_dist=False, overwrite=True)
if THING_TO_DO == 7:
    import numpy as np  # noqa
    from mayavi import mlab  # noqa
    from surfer import Brain  # noqa

    brain = Brain('sample', 'lh', 'inflated', subjects_dir=subjects_dir)
    surf = brain._geo

    vertidx = np.where(src[0]['inuse'])[0]

    mlab.points3d(surf.x[vertidx], surf.y[vertidx],
                          surf.z[vertidx], color=(1, 1, 0), scale_factor=1.5)
    raw_input("Press Enter to continue...")
    sys.exit(0)

bem_sol = mne.make_bem_solution(surfaces)
fwd_model = mne.make_forward_solution(info=evoked.info, src=src, bem=bem_sol, eeg=True, meg=False, trans=trans)

cov = mne.compute_covariance(epochs, tmax=0)
inv = mne.minimum_norm.make_inverse_operator(evoked.info, fwd_model, cov, loose=0.2)

stc = mne.minimum_norm.apply_inverse(evoked=evoked, inverse_operator=inv, lambda2=1./9., method='MNE')

if THING_TO_DO == 8:
    stc.plot(subjects_dir=subjects_dir)
