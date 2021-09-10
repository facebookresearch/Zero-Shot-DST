# Copyright (c) Facebook, Inc. and its affiliates
# All rights reserved.

# mostly from https://github.com/mrqa/MRQA-Shared-Task-2019

set -e
# mrqa train
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SQuAD.jsonl.gz -O qa_data/mrqa_train/SQuAD.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NewsQA.jsonl.gz -O qa_data/mrqa_train/NewsQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/TriviaQA-web.jsonl.gz -O qa_data/mrqa_train/TriviaQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/SearchQA.jsonl.gz -O qa_data/mrqa_train/SearchQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/HotpotQA.jsonl.gz -O qa_data/mrqa_train/HotpotQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/train/NaturalQuestionsShort.jsonl.gz -O qa_data/mrqa_train/NaturalQuestions.jsonl.gz
# mrqa valid
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SQuAD.jsonl.gz -O qa_data/mrqa_valid/SQuAD.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NewsQA.jsonl.gz -O qa_data/mrqa_valid/NewsQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/TriviaQA-web.jsonl.gz -O qa_data/mrqa_valid/TriviaQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/SearchQA.jsonl.gz -O qa_data/mrqa_valid/SearchQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/HotpotQA.jsonl.gz -O qa_data/mrqa_valid/HotpotQA.jsonl.gz
wget https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NaturalQuestionsShort.jsonl.gz -O qa_data/mrqa_valid/NaturalQuestions.jsonl.gz
# squad2
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O qa_data/squad2/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O qa_data/squad2/dev-v2.0.json
# dream
wget https://github.com/nlpdata/dream/raw/master/data/train.json -O qa_data/dream/train.json
wget https://github.com/nlpdata/dream/raw/master/data/dev.json -O qa_data/dream/dev.json
wget https://github.com/nlpdata/dream/raw/master/data/test.json -O qa_data/dream/test.json
