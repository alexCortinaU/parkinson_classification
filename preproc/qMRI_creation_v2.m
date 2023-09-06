%-----------------------------------------------------------------------
% Job saved on 13-Feb-2023 11:48:24 by cfg_util (rev $Rev: 7345 $)
% spm SPM - SPM12 (7771)
% cfg_basicio BasicIO - Unknown
%-----------------------------------------------------------------------
% subjects = [004 005 006 007 008 009 010 011 012 013];
% subjects = [015 016 017 018 019 020 021 022 023 024 025];
% subjects = [026 027 028 030];

% THIS COPY OF FILE IS FOR SUB 70 TO 74, THAT DO NOT HAVE THE 
% ses-01prisma3t_acq-hardiref_dir-PA_run-01_epi.nii.gz FILE IN THEIR
% ORIGINAL BIDS FOLDER
% subjects = [070 071 072 073 074];
subjects = [058 080];
for subject = subjects
    subj_no = num2str(subject, '%03d');
    subject = ['sub-' subj_no];

matlabbatch{1}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.parent = {'/mnt/scratch/7TPD/mpm_run_acu/bids'};
matlabbatch{1}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.name = subject;
matlabbatch{2}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.parent(1) = cfg_dep(['Make Directory: Make Directory' '''' subject ''''], substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','dir'));
matlabbatch{2}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.name = 'ses-01prisma3t';
matlabbatch{3}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.parent(1) = cfg_dep('Make Directory: Make Directory ''ses-01prisma3t''', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','dir'));
matlabbatch{3}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.name = 'anat';
matlabbatch{4}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.parent(1) = cfg_dep('Make Directory: Make Directory ''ses-01prisma3t''', substruct('.','val', '{}',{2}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','dir'));
matlabbatch{4}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.name = 'fmap';
%%
matlabbatch{5}.cfg_basicio.file_dir.file_ops.file_move.files = {
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude-acq-MToff_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude-acq-MTon_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude-acq-T1w_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-01_part-phase-acq-MToff_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-01_part-phase-acq-MTon_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-01_part-phase-acq-T1w_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude-acq-MToff_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude-acq-MTon_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude-acq-T1w_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-02_part-phase-acq-MToff_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-02_part-phase-acq-MTon_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-02_part-phase-acq-T1w_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-03_part-magnitude-acq-MToff_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-03_part-magnitude-acq-MTon_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-03_part-magnitude-acq-T1w_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-03_part-phase-acq-MToff_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-03_part-phase-acq-MTon_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-03_part-phase-acq-T1w_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-04_part-magnitude-acq-MToff_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-04_part-magnitude-acq-MTon_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-04_part-magnitude-acq-T1w_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-04_part-phase-acq-MToff_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-04_part-phase-acq-MTon_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-04_part-phase-acq-T1w_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-05_part-magnitude-acq-MToff_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-05_part-magnitude-acq-MTon_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-05_part-magnitude-acq-T1w_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-05_part-phase-acq-MToff_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-05_part-phase-acq-MTon_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-05_part-phase-acq-T1w_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-06_part-magnitude-acq-MToff_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-06_part-magnitude-acq-MTon_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-06_part-magnitude-acq-T1w_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-06_part-phase-acq-MToff_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-06_part-phase-acq-MTon_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-06_part-phase-acq-T1w_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-07_part-magnitude-acq-MToff_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-07_part-magnitude-acq-T1w_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-07_part-phase-acq-MToff_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-07_part-phase-acq-T1w_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-08_part-magnitude-acq-MToff_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-08_part-magnitude-acq-T1w_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-08_part-phase-acq-MToff_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-08_part-phase-acq-T1w_MPM.json']
                                                                };
%%
matlabbatch{5}.cfg_basicio.file_dir.file_ops.file_move.action.copyto(1) = cfg_dep('Make Directory: Make Directory ''anat''', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','dir'));
%%
matlabbatch{6}.cfg_basicio.file_dir.file_ops.file_move.files = {
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude-acq-B0_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-01-acq-B1_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-02-acq-B1_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-03-acq-B1_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-04-acq-B1_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-05-acq-B1_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-06-acq-B1_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-07-acq-B1_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-08-acq-B1_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-09-acq-B1_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-10-acq-B1_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-11-acq-B1_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude-acq-B0_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-01-acq-B1_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-02-acq-B1_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-03-acq-B1_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-04-acq-B1_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-05-acq-B1_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-06-acq-B1_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-07-acq-B1_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-08-acq-B1_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-09-acq-B1_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-10-acq-B1_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-11-acq-B1_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-phase-acq-B0_MPM.json']
                                                                ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_task-rest_dir-PA_epi.json']
                                                                };
%%
matlabbatch{6}.cfg_basicio.file_dir.file_ops.file_move.action.copyto(1) = cfg_dep('Make Directory: Make Directory ''fmap''', substruct('.','val', '{}',{4}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','dir'));
%%
matlabbatch{7}.cfg_basicio.file_dir.file_ops.cfg_gunzip_files.files = {
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude-acq-MToff_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude-acq-MTon_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude-acq-T1w_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-01_part-phase-acq-MToff_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-01_part-phase-acq-MTon_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-01_part-phase-acq-T1w_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude-acq-MToff_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude-acq-MTon_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude-acq-T1w_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-02_part-phase-acq-MToff_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-02_part-phase-acq-MTon_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-02_part-phase-acq-T1w_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-03_part-magnitude-acq-MToff_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-03_part-magnitude-acq-MTon_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-03_part-magnitude-acq-T1w_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-03_part-phase-acq-MToff_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-03_part-phase-acq-MTon_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-03_part-phase-acq-T1w_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-04_part-magnitude-acq-MToff_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-04_part-magnitude-acq-MTon_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-04_part-magnitude-acq-T1w_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-04_part-phase-acq-MToff_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-04_part-phase-acq-MTon_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-04_part-phase-acq-T1w_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-05_part-magnitude-acq-MToff_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-05_part-magnitude-acq-MTon_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-05_part-magnitude-acq-T1w_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-05_part-phase-acq-MToff_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-05_part-phase-acq-MTon_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-05_part-phase-acq-T1w_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-06_part-magnitude-acq-MToff_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-06_part-magnitude-acq-MTon_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-06_part-magnitude-acq-T1w_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-06_part-phase-acq-MToff_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-06_part-phase-acq-MTon_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-06_part-phase-acq-T1w_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-07_part-magnitude-acq-MToff_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-07_part-magnitude-acq-T1w_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-07_part-phase-acq-MToff_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-07_part-phase-acq-T1w_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-08_part-magnitude-acq-MToff_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-08_part-magnitude-acq-T1w_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-08_part-phase-acq-MToff_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-08_part-phase-acq-T1w_MPM.nii.gz']
                                                                       };
%%
matlabbatch{7}.cfg_basicio.file_dir.file_ops.cfg_gunzip_files.outdir(1) = cfg_dep('Make Directory: Make Directory ''anat''', substruct('.','val', '{}',{3}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','dir'));
matlabbatch{7}.cfg_basicio.file_dir.file_ops.cfg_gunzip_files.keep = true;
%%
matlabbatch{8}.cfg_basicio.file_dir.file_ops.cfg_gunzip_files.files = {                                                                       
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude-acq-B0_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-01-acq-B1_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-02-acq-B1_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-03-acq-B1_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-04-acq-B1_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-05-acq-B1_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-06-acq-B1_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-07-acq-B1_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-08-acq-B1_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-09-acq-B1_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-10-acq-B1_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-11-acq-B1_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude-acq-B0_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-01-acq-B1_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-02-acq-B1_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-03-acq-B1_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-04-acq-B1_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-05-acq-B1_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-06-acq-B1_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-07-acq-B1_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-08-acq-B1_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-09-acq-B1_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-10-acq-B1_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-11-acq-B1_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-phase-acq-B0_MPM.nii.gz']
                                                                       ['/mnt/projects/7TPD/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_task-rest_dir-PA_epi.nii.gz']
                                                                       };
%%
matlabbatch{8}.cfg_basicio.file_dir.file_ops.cfg_gunzip_files.outdir(1) = cfg_dep('Make Directory: Make Directory ''fmap''', substruct('.','val', '{}',{4}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','dir'));
matlabbatch{8}.cfg_basicio.file_dir.file_ops.cfg_gunzip_files.keep = true;
matlabbatch{9}.spm.spatial.preproc.channel.vols = {['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude-acq-T1w_MPM.nii,1']};
matlabbatch{9}.spm.spatial.preproc.channel.biasreg = 0.001;
matlabbatch{9}.spm.spatial.preproc.channel.biasfwhm = 60;
matlabbatch{9}.spm.spatial.preproc.channel.write = [1 1];
matlabbatch{9}.spm.spatial.preproc.tissue(1).tpm = {'/home/alejandrocu/spm12/tpm/TPM.nii,1'};
matlabbatch{9}.spm.spatial.preproc.tissue(1).ngaus = 1;
matlabbatch{9}.spm.spatial.preproc.tissue(1).native = [1 0];
matlabbatch{9}.spm.spatial.preproc.tissue(1).warped = [0 0];
matlabbatch{9}.spm.spatial.preproc.tissue(2).tpm = {'/home/alejandrocu/spm12/tpm/TPM.nii,2'};
matlabbatch{9}.spm.spatial.preproc.tissue(2).ngaus = 1;
matlabbatch{9}.spm.spatial.preproc.tissue(2).native = [1 0];
matlabbatch{9}.spm.spatial.preproc.tissue(2).warped = [0 0];
matlabbatch{9}.spm.spatial.preproc.tissue(3).tpm = {'/home/alejandrocu/spm12/tpm/TPM.nii,3'};
matlabbatch{9}.spm.spatial.preproc.tissue(3).ngaus = 2;
matlabbatch{9}.spm.spatial.preproc.tissue(3).native = [1 0];
matlabbatch{9}.spm.spatial.preproc.tissue(3).warped = [0 0];
matlabbatch{9}.spm.spatial.preproc.tissue(4).tpm = {'/home/alejandrocu/spm12/tpm/TPM.nii,4'};
matlabbatch{9}.spm.spatial.preproc.tissue(4).ngaus = 3;
matlabbatch{9}.spm.spatial.preproc.tissue(4).native = [1 0];
matlabbatch{9}.spm.spatial.preproc.tissue(4).warped = [0 0];
matlabbatch{9}.spm.spatial.preproc.tissue(5).tpm = {'/home/alejandrocu/spm12/tpm/TPM.nii,5'};
matlabbatch{9}.spm.spatial.preproc.tissue(5).ngaus = 4;
matlabbatch{9}.spm.spatial.preproc.tissue(5).native = [1 0];
matlabbatch{9}.spm.spatial.preproc.tissue(5).warped = [0 0];
matlabbatch{9}.spm.spatial.preproc.tissue(6).tpm = {'/home/alejandrocu/spm12/tpm/TPM.nii,6'};
matlabbatch{9}.spm.spatial.preproc.tissue(6).ngaus = 2;
matlabbatch{9}.spm.spatial.preproc.tissue(6).native = [1 0];
matlabbatch{9}.spm.spatial.preproc.tissue(6).warped = [0 0];
matlabbatch{9}.spm.spatial.preproc.warp.mrf = 1;
matlabbatch{9}.spm.spatial.preproc.warp.cleanup = 1;
matlabbatch{9}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
matlabbatch{9}.spm.spatial.preproc.warp.affreg = 'mni';
matlabbatch{9}.spm.spatial.preproc.warp.fwhm = 0;
matlabbatch{9}.spm.spatial.preproc.warp.samp = 3;
matlabbatch{9}.spm.spatial.preproc.warp.write = [0 0];
matlabbatch{9}.spm.spatial.preproc.warp.vox = NaN;
matlabbatch{9}.spm.spatial.preproc.warp.bb = [NaN NaN NaN
                                              NaN NaN NaN];
matlabbatch{10}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.parent = {'/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI'};
matlabbatch{10}.cfg_basicio.file_dir.dir_ops.cfg_mkdir.name = subject;
matlabbatch{11}.spm.tools.hmri.autoreor.reference = {['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/msub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude-acq-T1w_MPM.nii,1']};
matlabbatch{11}.spm.tools.hmri.autoreor.template = {'/home/alejandrocu/spm12/canonical/avg152T1.nii'};
%%
matlabbatch{11}.spm.tools.hmri.autoreor.other = {
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude-acq-MToff_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude-acq-MTon_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude-acq-T1w_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-01_part-phase-acq-MToff_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-01_part-phase-acq-MTon_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-01_part-phase-acq-T1w_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude-acq-MToff_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude-acq-MTon_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude-acq-T1w_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-02_part-phase-acq-MToff_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-02_part-phase-acq-MTon_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-02_part-phase-acq-T1w_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-03_part-magnitude-acq-MToff_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-03_part-magnitude-acq-MTon_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-03_part-magnitude-acq-T1w_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-03_part-phase-acq-MToff_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-03_part-phase-acq-MTon_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-03_part-phase-acq-T1w_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-04_part-magnitude-acq-MToff_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-04_part-magnitude-acq-MTon_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-04_part-magnitude-acq-T1w_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-04_part-phase-acq-MToff_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-04_part-phase-acq-MTon_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-04_part-phase-acq-T1w_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-05_part-magnitude-acq-MToff_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-05_part-magnitude-acq-MTon_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-05_part-magnitude-acq-T1w_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-05_part-phase-acq-MToff_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-05_part-phase-acq-MTon_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-05_part-phase-acq-T1w_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-06_part-magnitude-acq-MToff_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-06_part-magnitude-acq-MTon_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-06_part-magnitude-acq-T1w_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-06_part-phase-acq-MToff_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-06_part-phase-acq-MTon_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-06_part-phase-acq-T1w_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-07_part-magnitude-acq-MToff_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-07_part-magnitude-acq-T1w_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-07_part-phase-acq-MToff_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-07_part-phase-acq-T1w_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-08_part-magnitude-acq-MToff_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-08_part-magnitude-acq-T1w_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-08_part-phase-acq-MToff_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/anat/sub-' subj_no '_ses-01prisma3t_echo-08_part-phase-acq-T1w_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude-acq-B0_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-01-acq-B1_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-02-acq-B1_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-03-acq-B1_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-04-acq-B1_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-05-acq-B1_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-06-acq-B1_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-07-acq-B1_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-08-acq-B1_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-09-acq-B1_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-10-acq-B1_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-11-acq-B1_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude-acq-B0_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-01-acq-B1_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-02-acq-B1_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-03-acq-B1_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-04-acq-B1_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-05-acq-B1_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-06-acq-B1_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-07-acq-B1_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-08-acq-B1_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-09-acq-B1_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-10-acq-B1_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-11-acq-B1_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_echo-02_part-phase-acq-B0_MPM.nii,1']
                                                 ['/mnt/scratch/7TPD/mpm_run_acu/bids/' subject '/ses-01prisma3t/fmap/sub-' subj_no '_ses-01prisma3t_task-rest_dir-PA_epi.nii,1']
                                                 };
%%
matlabbatch{11}.spm.tools.hmri.autoreor.output.outdir(1) = cfg_dep(['Make Directory: Make Directory' '''' subject ''''], substruct('.','val', '{}',{10}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','dir'));
matlabbatch{11}.spm.tools.hmri.autoreor.dep = 'individual';
matlabbatch{12}.spm.tools.hmri.hmri_config.hmri_setdef.customised = {'/home/alejandrocu/hMRI-toolbox-0.5.0/config/local/hmri_local_defaults_w_WLS1.m'};
matlabbatch{13}.spm.tools.hmri.create_mpm.subj.output.outdir = {['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject]};
matlabbatch{13}.spm.tools.hmri.create_mpm.subj.sensitivity.RF_us = '-';
%%
matlabbatch{13}.spm.tools.hmri.create_mpm.subj.b1_type.i3D_EPI.b1input = {
                                                                          ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-01-acq-B1_MPM.nii,1']
                                                                          ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-01-acq-B1_MPM.nii,1']
                                                                          ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-02-acq-B1_MPM.nii,1']
                                                                          ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-02-acq-B1_MPM.nii,1']
                                                                          ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-03-acq-B1_MPM.nii,1']
                                                                          ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-03-acq-B1_MPM.nii,1']
                                                                          ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-04-acq-B1_MPM.nii,1']
                                                                          ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-04-acq-B1_MPM.nii,1']
                                                                          ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-05-acq-B1_MPM.nii,1']
                                                                          ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-05-acq-B1_MPM.nii,1']
                                                                          ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-06-acq-B1_MPM.nii,1']
                                                                          ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-06-acq-B1_MPM.nii,1']
                                                                          ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-07-acq-B1_MPM.nii,1']
                                                                          ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-07-acq-B1_MPM.nii,1']
                                                                          ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-08-acq-B1_MPM.nii,1']
                                                                          ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-08-acq-B1_MPM.nii,1']
                                                                          ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-09-acq-B1_MPM.nii,1']
                                                                          ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-09-acq-B1_MPM.nii,1']
                                                                          ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-10-acq-B1_MPM.nii,1']
                                                                          ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-10-acq-B1_MPM.nii,1']
                                                                          ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude_fa-11-acq-B1_MPM.nii,1']
                                                                          ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude_fa-11-acq-B1_MPM.nii,1']
                                                                          };
%%
matlabbatch{13}.spm.tools.hmri.create_mpm.subj.b1_type.i3D_EPI.b0input = {
                                                                          ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude-acq-B0_MPM.nii,1']
                                                                          ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude-acq-B0_MPM.nii,1']
                                                                          ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-02_part-phase-acq-B0_MPM.nii,1']
                                                                          };
matlabbatch{13}.spm.tools.hmri.create_mpm.subj.b1_type.i3D_EPI.b1parameters.b1metadata = 'yes';
matlabbatch{13}.spm.tools.hmri.create_mpm.subj.raw_mpm.MT = {
                                                             ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude-acq-MTon_MPM.nii,1']
                                                             ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude-acq-MTon_MPM.nii,1']
                                                             ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-03_part-magnitude-acq-MTon_MPM.nii,1']
                                                             ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-04_part-magnitude-acq-MTon_MPM.nii,1']
                                                             ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-05_part-magnitude-acq-MTon_MPM.nii,1']
                                                             ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-06_part-magnitude-acq-MTon_MPM.nii,1']
                                                             };
matlabbatch{13}.spm.tools.hmri.create_mpm.subj.raw_mpm.PD = {
                                                             ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude-acq-MToff_MPM.nii,1']
                                                             ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude-acq-MToff_MPM.nii,1']
                                                             ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-03_part-magnitude-acq-MToff_MPM.nii,1']
                                                             ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-04_part-magnitude-acq-MToff_MPM.nii,1']
                                                             ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-05_part-magnitude-acq-MToff_MPM.nii,1']
                                                             ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-06_part-magnitude-acq-MToff_MPM.nii,1']
                                                             ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-07_part-magnitude-acq-MToff_MPM.nii,1']
                                                             ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-08_part-magnitude-acq-MToff_MPM.nii,1']
                                                             };
matlabbatch{13}.spm.tools.hmri.create_mpm.subj.raw_mpm.T1 = {
                                                             ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-01_part-magnitude-acq-T1w_MPM.nii,1']
                                                             ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-02_part-magnitude-acq-T1w_MPM.nii,1']
                                                             ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-03_part-magnitude-acq-T1w_MPM.nii,1']
                                                             ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-04_part-magnitude-acq-T1w_MPM.nii,1']
                                                             ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-05_part-magnitude-acq-T1w_MPM.nii,1']
                                                             ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-06_part-magnitude-acq-T1w_MPM.nii,1']
                                                             ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-07_part-magnitude-acq-T1w_MPM.nii,1']
                                                             ['/mnt/scratch/7TPD/mpm_run_acu/bids/derivatives/hMRI/' subject '/AutoReorient/sub-' subj_no '_ses-01prisma3t_echo-08_part-magnitude-acq-T1w_MPM.nii,1']
                                                             };
matlabbatch{13}.spm.tools.hmri.create_mpm.subj.popup = false;

spm_jobman('run', matlabbatch)
end