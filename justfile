default:
    just --list

train:
    python main.py --dataset_path="../ribseg_benchmark"

train_binary:
    python main.py --binary --dataset_path="../ribseg_benchmark"

train_binary_second_stage:
    python main.py --dataset_path="/data/adhinart/ribseg/outputs/pointcnn" --binary_dataset_path="/data/adhinart/ribseg/outputs/pointcnn_binary"

inference:
    python inference.py --model_path="/data/adhinart/ribseg/pointcnn_pytorch/lightning_logs/1b6lw16i/checkpoints/epoch=199-step=600.ckpt" --output_dir="/data/adhinart/ribseg/pointcnn_pytorch/outputs/normal" --dataset_path="../ribseg_benchmark"

inference_binary:
    python inference.py --binary --model_path="/data/adhinart/ribseg/pointcnn_pytorch/lightning_logs/g4ufgb1l/checkpoints/epoch=199-step=600.ckpt" --output_dir="/data/adhinart/ribseg/pointcnn_pytorch/outputs/binary" --dataset_path="../ribseg_benchmark"

# note fix this
inference_binary_second_stage:
    python inference.py --model_path="/data/adhinart/ribseg/pointcnn_pytorch/lightning_logs/cwwkx4tw/checkpoints/epoch=199-step=600.ckpt" --output_dir="/data/adhinart/ribseg/pointcnn_pytorch/outputs/second_stage" --dataset_path="/data/adhinart/ribseg/outputs/pointcnn" --binary_dataset_path="/data/adhinart/ribseg/outputs/pointcnn_binary"

time:
    python inference.py --model_path="/data/adhinart/ribseg/pointcnn_pytorch/lightning_logs/1b6lw16i/checkpoints/epoch=199-step=600.ckpt" --output_dir="/data/adhinart/ribseg/pointcnn_pytorch/outputs/normal" --dataset_path="../ribseg_benchmark" --dry_run=10 --batch_size=16

time_binary:
    python inference.py --binary --model_path="/data/adhinart/ribseg/pointcnn_pytorch/lightning_logs/g4ufgb1l/checkpoints/epoch=199-step=600.ckpt" --output_dir="/data/adhinart/ribseg/pointcnn_pytorch/outputs/binary" --dataset_path="../ribseg_benchmark" --dry_run=10 --batch_size=16

# note fix this
time_binary_second_stage:
    python inference.py --model_path="/data/adhinart/ribseg/pointcnn_pytorch/lightning_logs/cwwkx4tw/checkpoints/epoch=199-step=600.ckpt" --output_dir="/data/adhinart/ribseg/pointcnn_pytorch/outputs/second_stage" --dataset_path="/data/adhinart/ribseg/outputs/pointcnn" --binary_dataset_path="/data/adhinart/ribseg/outputs/pointcnn_binary" --dry_run=10 --batch_size=16
