mpiexec -np 8 --oversubscribe --allow-run-as-root python3 /nfs/home/dstojanovski/docker_files/image_train.py --datadir "/nfs/home/dstojanovski/docker_files/fresh_data/512_data/2CH_combo" --savedir "./output_2ch_combo_512" --batch_size_train 12 --is_train True --save_interval 50000 --lr_anneal_steps 100001 --random_flip True --deterministic_train False --img_size 512 --microbatch 4

