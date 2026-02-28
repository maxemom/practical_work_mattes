# Practical_Work_Mattes (PWM)

docker build -t pwm .

mkdir -p outputs
docker run --rm -it \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/outputs:/workspace/outputs" \
  pwm bash

  python scripts/prepare_prompts.py --input data/raw/wiki.txt --output data/prompts.txt
python run_grid.py --base configs/base.yaml --grid configs/grid.yaml

docker run --rm \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/outputs:/workspace/outputs" \
  pwm python run_grid.py --base configs/base.yaml --grid configs/grid.yaml