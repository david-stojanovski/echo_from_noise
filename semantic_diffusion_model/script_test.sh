mpiexec -np 8 --oversubscribe --allow-run-as-root python3 /nfs/home/dstojanovski/docker_files/image_sample.py --datadir '/nfs/home/dstojanovski/docker_files/fresh_data/256_data/BB_2CH_ED' --resume_checkpoint '/nfs/home/dstojanovski/docker_files/ema_0.9999_050000_2ch_ed.pt' --batch_size_test 5 --results_dir '/nfs/home/dstojanovski/docker_files/results_biobank_2CH_ED' --num_samples 4113 --is_train False --inference_on_train True







