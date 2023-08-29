# training of the model
python train.py cifar10 cifar_resnet110 --lr 0.01 --lr_step_size 50 --epochs 150  --noise 0.50 --num_noise_vec 2 --lbd 20 --k 1530 --k_warmup 100 --save_path cifar10_half_consistency_150_001.pth

## Neyman Pearson
# python sampler.py cifar10 dsrs/hamare_models/cifar10_half_consistency_150_001.pth 0.50 --disttype general-gaussian --k 1530 --N 100000 --alpha 0.001 --skip 10 --batch 400 
# python main.py cifar10 origin cifar10_half_consistency_150_001 general-gaussian --k 1530 0.50 100000 0.001 --workers 20

# ## DSRS Certification
python sampler.py cifar10 consistency-cifar-0.50.pth.tar 0.50 --disttype general-gaussian --k 1530 --N 1 --alpha 0.0005 --skip 10 --batch 400 
python sampler.py cifar10 consistency-cifar-0.50.pth.tar 0.4 --disttype general-gaussian --k 1530 --N 1 --alpha 0.0005 --skip 10 --batch 400 
python main.py cifar10 origin consistency-cifar-0.50.pth general-gaussian --k 1530 0.50 50000 0.0005 --workers 20
python main.py cifar10 improved consistency-cifar-0.50.pth general-gaussian --k 1530 0.50 50000 0.0005 -b 0.4 --workers 20