# from transformers import pipeline

# generator = pipeline("text-generation", "./gpt2_test")

# print(generator("hello", ))

# see inference in
# https://huggingface.co/docs/transformers/en/tasks/language_modeling
# from transformers import AutoModelForCausalLM, AutoTokenizer

# model = AutoModelForCausalLM.from_pretrained("./gpt2_test")
# tokenizer = AutoTokenizer.from_pretrained("./gpt2_test")

# # encode context the generation is conditioned on
# input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors="pt")

# # generate text until the output length (which includes the context length) reaches 50
# greedy_output = model(input_ids)

# print(greedy_output)

# print("Output:\n" + 100 * '-')
# print(tokenizer.decode(greedy_output, skip_special_tokens=True))

# exit()
# First, get some data; put in dummy data folder
# https://github.com/huggingface/blog/blob/main/notebooks/01_how_to_train.ipynb

# ###############################################
# second, train a tokenizer on the dummy data
# API has changed quite a bit; this way is working
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# download some text data
# wget -c https://cdn-datasets.huggingface.co/EsperBERTo/data/oscar.eo.txt
# the oscar data file is too large; lets copy the first 100K bytes
# head -c 100000 oscar.eo.txt > 100K_oscar.eo.txt

# Don't know what special tokens are. But provide tokenizer some info about
# the text. Maybe <s> is space " "

train_tokenizer =  True
if train_tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tkn_trainer = BpeTrainer(special_tokens=["<s>", 
                                            "<pad>", 
                                            "</s>",
                                            "<mask>"])

    tokenizer.train(["dummy_data/100K_oscar.eo.txt"], tkn_trainer)

    # save the tokenizer
    tokenizer.save("custom_gpt2/tokenizer.json")
else:
    # load tokenizer
    from transformers import PreTrainedTokenizerBase
    tokenizer = PreTrainedTokenizerBase(tokenizer_file="custom_gpt2/tokenizer.json")

# ###############################################
exit()

# ###############################################
# third, build model
from transformers import GPT2Config, GPT2Model

# lots of options; but I am only reducing number of layers
gpt2_config = GPT2Config(num_layers=2)
gpt2_model = GPT2Model(gpt2_config)

# MB assuming fp32
print(gpt2_model.num_parameters() / 1e6, "M")

print(gpt2_model.get_memory_footprint() / (1024 * 1024), "MB")

# model size is fairly large; maybe need to go smaller; see other configs
# ################################################


# ###############################################
# fourth, dataset prep; the quick way
# https://github.com/huggingface/transformers/blob/936b57158ad2641390422274fed6ee6c2a685e15/examples/pytorch/language-modeling/run_mlm.py#L242
# https://discuss.huggingface.co/t/help-understanding-how-to-build-a-dataset-for-language-as-with-the-old-textdataset/5870/2

from datasets import load_dataset
dataset = load_dataset("text", data_files={"train": ["dummy_data/100K_oscar.eo.txt"], 
                                           "test": ["dummy_data/100K_oscar.eo.txt"]})

# This is just a small helper that will help us batch different samples 
# of the dataset together into an object that PyTorch knows how to perform backprop on.
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, mlm_probability=0.15
)
# ###############################################

# ###############################################
# fifth, dataset prep; the quick way

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./dummy_out",
    overwrite_output_dir=True,
    num_train_epochs=1,
    # per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=gpt2_model,
    args=training_args,
    data_collator=data_collator,
    dataset_text_field="text",
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

trainer.train()

##### OR #####
# see readme for clm train script
# https://github.com/huggingface/transformers/blob/936b57158ad2641390422274fed6ee6c2a685e15/examples/pytorch/language-modeling/README.md
#
# alternatively, newer version is available at
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
"""
python run_clm.py \
--model_name_or_path gpt2 \
--train_file dummy_data/5000_oscar.eo.txt \
--validation_file dummy_data/5000_oscar.eo.txt \
--do_train \
--do_eval \
--config_name /Users/saideeptiku/Projects/hf/gpt2_config.json \
--output_dir ./dummy_out


"""