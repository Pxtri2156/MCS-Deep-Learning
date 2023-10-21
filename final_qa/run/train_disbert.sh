python train.py \
    --train_path='/workspace/tripx/MCS/deep_learning/final_qa/data/nq-sub2-train-v1.0.1.json' \
    --dev_path='/workspace/tripx/MCS/deep_learning/final_qa/data/nq-train-v1.0.1.json' \
    --save_path='/workspace/tripx/MCS/deep_learning/final_qa/model/distibert' \
    --batch_size=16 \
    --epochs=5 \
    --model="distilbert-base-uncased" \
    --tokenizer="distilbert-base-uncased"