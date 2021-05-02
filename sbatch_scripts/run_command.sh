python run.py -data ./datasets -dataset-name stl10 --log-every-n-steps 100 --epochs 200  --arch resnet18 --out_dim 32 &
# python run.py -b 128 -data ./datasets -dataset-name stl10 --log-every-n-steps 100 --epochs 200  --arch resnet18 --out_dim 128 &
# python run.py -b 64 -data ./datasets -dataset-name stl10 --log-every-n-steps 100 --epochs 200  --arch resnet18 --out_dim 128 &
# python run.py -b 32 -data ./datasets -dataset-name stl10 --log-every-n-steps 100 --epochs 200  --arch resnet18 --out_dim 128 &
python run.py -data ./datasets -dataset-name stl10 --log-every-n-steps 100 --epochs 200  --arch resnet18 --out_dim 64 &
python run.py -data ./datasets -dataset-name stl10 --log-every-n-steps 100 --epochs 200  --arch resnet18 --out_dim 128 &
python run.py -data ./datasets -dataset-name stl10 --log-every-n-steps 100 --epochs 200  --arch resnet18 --out_dim 256 &
python run.py -data ./datasets -dataset-name cifar10 --log-every-n-steps 100 --epochs 200  --arch resnet18 --out_dim 128 &
python run.py -data ./datasets -dataset-name stl10 --log-every-n-steps 100 --epochs 200  --arch resnet50 --out_dim 128 &
