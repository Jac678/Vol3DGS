lego_datapath=../volumetric_gaussian_splatting/data/lego/

python3 train_volr.py -s $lego_datapath -m output/vol3dgs_lego --render_backend slang_volr --eval --densify_from_iter 50000 --disable_opacity_reset --iterations 7000
python3 train_volr.py -s $lego_datapath -m output/3dgs_lego --render_backend slang --eval --densify_from_iter 50000 --iterations 7000