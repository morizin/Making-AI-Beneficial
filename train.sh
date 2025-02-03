
python3 train.py --weight_dir "$1/v1/" -d "$2" --type 1 --epochs 70 --batch_size 256
python test.py --weight "$1/v1/sakt_model.pt" -d "$2" -o "$1/sub_sakt_model_v1.csv" --type 1 --batch_size 256
python3 train.py --weight_dir "$1/v1_pseudo/" -d "$2" --type 1 --epochs 70 --batch_size 256 --pseudo "$1/sub_sakt_model_v1.csv"

python3 train.py --weight_dir "$1/v2/" -d "$2" --type 2 --epochs 70 --batch_size 256
python test.py --weight "$1/v2/sakt_model.pt" -d "$2" -o "$1/sub_sakt_model_v2.csv" --type 2 --batch_size 256
python3 train.py --weight_dir "$1/v2_pseudo/" -d "$2" --type 2 --epochs 70 --batch_size 256 --pseudo "$1/sub_sakt_model_v2.csv"

python3 train.py --weight_dir "$1/v3/" -d "$2" --type 3 --epochs 70 --batch_size 256

python3 train.py --weight_dir "$1/v5/" -d "$2" --type 5 --epochs 70 --batch_size 256
python test.py --weight "$1/v5/sakt_model.pt" -d "$2" -o "$1/sub_sakt_model_v5.csv" --type 5 --batch_size 256
python3 train.py --weight_dir "$1/v5_pseudo/" -d "$2" --type 5 --epochs 70 --batch_size 256 --pseudo "$1/sub_sakt_model_v5.csv"

python3 train.py --weight_dir "$1/v6/" -d "$2" --type 6 --epochs 70 --batch_size 256
python test.py --weight "$1/v6/sakt_model.pt" -d "$2" -o "$1/sub_sakt_model_v6.csv" --type 6 --batch_size 256
python3 train.py --weight_dir "$1/v6_pseudo/" -d "$2" --type 6 --epochs 70 --batch_size 256 --pseudo "$1/sub_sakt_model_v6.csv"
python3 train.py --weight_dir "$1/v6_with_adam/" -d "$2" --type 6 --epochs 70 --batch_size 256 --adam

python3 train.py --weight_dir "$1/v7/" -d "$2" --type 7 --epochs 70 --batch_size 256
