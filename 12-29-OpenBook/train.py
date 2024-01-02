# 合并的数据
# https://www.kaggle.com/datasets/cdeotte/60k-data-with-context-v2/data
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from dataclasses import dataclass
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForMultipleChoice
import warnings
from typing import Optional, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
warnings.filterwarnings("ignore")


option_to_index = {option: idx for idx, option in enumerate('ABCDE')}
index_to_option = {v: k for k,v in option_to_index.items()}


class CFG:
    train_path = "datasets/all_12_with_context2.csv"
    valid_path = "datasets/train_with_context2.csv"
    NUM_TRAIN_SAMPLES = 1024
    MODEL = "/mnt/HDD0/wuzhiqiang/competition/FT-Data-Ranker/llm-se/pretrain-model/deberta-v3-large"
    MAX_INPUT = 256
    USE_PEFT = False
    FREEZE_EMBEDDINGS = True
    FREEZE_LAYERS = 18  # deberta总共有24层


def preprocess(example, tokenizer):
    first_sentence = ["[CLS] " + example['context']] * 5
    second_sentences = [" #### " + example['prompt'] + " [SEP] " + example[option] + " [SEP]" for option in 'ABCDE']
    tokenized_example = tokenizer(first_sentence, second_sentences, truncation='only_first', 
                                  max_length=CFG.MAX_INPUT, add_special_tokens=False)
    tokenized_example['label'] = option_to_index[example['answer']]
    return tokenized_example


# https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/435602
def map_at_3(predictions, labels):
    map_sum = 0
    pred = np.argsort(-1 * np.array(predictions), axis=1)[:,:3]
    for x, y in zip(pred, labels):
        z = [1 / i if y==j else 0 for i, j in zip([1,2,3], x)]
        map_sum += np.sum(z)
    return map_sum / len(predictions)

# Define your custom evaluation function
def compute_metrics(p):
    predictions = p.predictions.tolist()
    labels = p.label_ids.tolist()
    return {"map@3": map_at_3(predictions, labels)}


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features):
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch


def main():
    df_train = pd.read_csv(CFG.train_path)
    df_valid = pd.read_csv(CFG.valid_path)
    df_train = df_train.drop(columns="source")
    df_train = df_train.fillna('').sample(CFG.NUM_TRAIN_SAMPLES)
    print('Train data size:', df_train.shape)

    tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL)
    valid_dataset = Dataset.from_pandas(df_valid)
    train_dataset = Dataset.from_pandas(df_train)
    train_dataset = train_dataset.remove_columns(["__index_level_0__"])
    

    valid_dataset_tokenized = valid_dataset.map(preprocess, fn_kwargs={"tokenizer": tokenizer}, remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])
    train_dataset_tokenized = train_dataset.map(preprocess, fn_kwargs={"tokenizer": tokenizer}, remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E', 'answer'])
    
    print(train_dataset_tokenized)
    # Dataset({                                                                                                                                                          
    #     features: ['input_ids', 'token_type_ids', 'attention_mask', 'label'],
    #     num_rows: 1024
    # })

    model = AutoModelForMultipleChoice.from_pretrained(CFG.MODEL)

    if CFG.USE_PEFT:
        print("using peft")
        from peft import LoraConfig, get_peft_model, TaskType
        peft_config = LoraConfig(
            r=8,
            lora_alpha=4,
            task_type=TaskType.SEQ_CLS,
            lora_dropout=0.1,
            bias="none",
            inference_mode=False,
            target_modules=["query_proj", "value_proj"],
            modules_to_save=["classifier", "pooler"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    if CFG.FREEZE_EMBEDDINGS:
        # 将embedding层固定住
        print('Freezing embeddings.')
        for param in model.deberta.embeddings.parameters():
            param.requires_grad = False
         
    if CFG.FREEZE_LAYERS > 0:
        print(f'Freezing {CFG.FREEZE_LAYERS} layers.')
        for layer in model.deberta.encoder.layer[: CFG.FREEZE_LAYERS]:
            for param in layer.parameters():
                param.requires_grad = False


    training_args = TrainingArguments(
        warmup_ratio=0.1, 
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        num_train_epochs=4,
        report_to='none',
        output_dir = f'./checkpoints',
        overwrite_output_dir=True,
        fp16=True,
        gradient_accumulation_steps=8,
        logging_steps=25,
        evaluation_strategy='steps',
        eval_steps=25,
        save_strategy="steps",
        save_steps=25,
        load_best_model_at_end=False,
        metric_for_best_model='map@3',
        lr_scheduler_type='cosine',
        weight_decay=0.01,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        train_dataset=train_dataset_tokenized,
        eval_dataset=valid_dataset_tokenized,
        compute_metrics = compute_metrics,
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
    trainer.save_model(f'model')


# https://www.kaggle.com/code/philippsinger/h2ogpt-perplexity-ranking
import numpy as np
def precision_at_k(r, k):
    """Precision at k"""
    assert k <= len(r)
    assert k != 0
    return sum(int(x) for x in r[:k]) / k


def MAP_at_3(predictions, true_items):
    """Score is mean average precision at 3"""
    U = len(predictions)
    map_at_3 = 0.0
    for u in range(U):
        user_preds = predictions[u].split()
        user_true = true_items[u]
        user_results = [1 if item == user_true else 0 for item in user_preds]
        for k in range(min(len(user_preds), 3)):
            map_at_3 += precision_at_k(user_results, k+1) * user_results[k]
    return map_at_3 / U


def evaluate():
    if CFG.USE_PEFT:
        print("using peft")
        from peft import LoraConfig, get_peft_model, TaskType

        model = AutoModelForMultipleChoice.from_pretrained(CFG.MODEL)
        peft_config = LoraConfig(
            r=8,
            lora_alpha=4,
            task_type=TaskType.SEQ_CLS,
            lora_dropout=0.1,
            bias="none",
            inference_mode=False,
            target_modules=["query_proj", "value_proj"],
            modules_to_save=["classifier", "pooler"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        checkpoint = torch.load(f'model/pytorch_model.bin')
        model.load_state_dict(checkpoint)
    else:
        model = AutoModelForMultipleChoice.from_pretrained(f'model')

    trainer = Trainer(model=model)
    tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL)

    test_df = pd.read_csv(CFG.valid_path)
    test_dataset_tokenized = Dataset.from_pandas(test_df).map(
        preprocess, fn_kwargs={"tokenizer": tokenizer}, remove_columns=['prompt', 'context', 'A', 'B', 'C', 'D', 'E'])

    # 对验证集测试计算map@3
    test_predictions = trainer.predict(test_dataset_tokenized).predictions
    predictions_as_ids = np.argsort(-test_predictions, 1)
    predictions_as_answer_letters = np.array(list('ABCDE'))[predictions_as_ids]
    test_df['prediction'] = [
        ' '.join(row) for row in predictions_as_answer_letters[:, :3]
        ]
    m = MAP_at_3(test_df.prediction.values, test_df.answer.values)
    print( 'CV MAP@3 =', m)

if __name__ == "__main__":
    main()
    evaluate()