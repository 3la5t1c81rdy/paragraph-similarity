---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:4
- loss:SEF1MSE
- loss:MSELoss
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer

This is a [sentence-transformers](https://www.SBERT.net) model trained. It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
- **Maximum Sequence Length:** 384 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 384, 'do_lower_case': False, 'architecture': 'MPNetModel'})
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'The weather is lovely today.',
    "It's so sunny outside!",
    'He drove to the stadium.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.8525, 0.0352],
#         [0.8525, 1.0000, 0.0396],
#         [0.0352, 0.0396, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 4 training samples
* Columns: <code>paragraphs1</code>, <code>paragraphs2</code>, and <code>label</code>
* Approximate statistics based on the first 4 samples:
  |         | paragraphs1                                                                         | paragraphs2                                                                          | label                                           |
  |:--------|:------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:------------------------------------------------|
  | type    | string                                                                              | string                                                                               | int                                             |
  | details | <ul><li>min: 46 tokens</li><li>mean: 222.0 tokens</li><li>max: 384 tokens</li></ul> | <ul><li>min: 60 tokens</li><li>mean: 222.25 tokens</li><li>max: 384 tokens</li></ul> | <ul><li>0: ~50.00%</li><li>1: ~50.00%</li></ul> |
* Samples:
  | paragraphs1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | paragraphs2                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | label          |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------|
  | <code>The lessons are short and the order varied, but in one single<br>morning or afternoon there is a bewildering number of changes. Some years ago the unfortunate<br>principle was laid down in the Code, that fifteen minutes was sufficient time for a lesson<br>in an Infant School, and though this is not strictly followed the lessons are short and numerous,<br>giving an unsettled character to the work; children appear to be swung at a moment's notice<br>from topic to topic without an apparent link or reason: for example, the day's work may begin<br>with the story of a little boy sent by train to the country, settled at a farm and taken out<br>to see the cow and the sow: soon this is found to be a reading lesson on words ending in "ow,"<br>but after a short time the whole class is told quite suddenly, that one shilling is to be spent<br>at a shop in town, and while they are still interested in calculating the change, paints are<br>distributed, and the children are painting the bluebell. The whole day is apt to be of this<br>...</code> | <code>Although the lessons are short and varied, in a single morning or after noon there are many changes. Fifteen minutes was believed to be sufficient time for a lesson in an Infant School some years ago. This is not strictly followed.  Lessons become short and numerous, and it gives an unsettled character to the work;  At a moments notice children are swung from topic to topic with no well founded link or reason.   For example, the day may begin with the story of a boy sent by train to the country, or settled at a farm and taken out to see the cow and the sow.   But after a while this is found to be a reading lesson on words ending in "ow," but after the whole class is told quite suddenly after a short time, that one is to used in town at a shop.  And they still calculate change in the distributed paints.  And the children are painting the blueblee.   Certainly the whole day is sure to be of this broken character. And it doesn't make for training in mental concentration or for unity in th...</code>                                              | <code>1</code> |
  | <code>But she discovered<br>it by accident, and without declaring any such intention, she gave up her pen and her books,<br>and applied herself exclusively to household business, for several months, till her body as<br>well as her spirits failed. She became emaciated, her countenance bore marks of deep dejection,<br>and often, while actively employed in domestic duties, she could neither restrain nor conceal<br>her tears. The mother seems to have been slower in perceiving this than she would have been<br>had it not been for her own state of confinement; she noticed it at length, and said, "Lucretia,<br>it is a long time since you have written any thing." The girl then burst into tears, and replied,<br>"O mother, I have given that up long ago." "But why?" said her mother. After much emotion,<br>she answered, "I am convinced from what my friends have said, and from what I see, that I have<br>done wrong in pursuing the course I have. I well know the circumstances of the family are such,<br>that it requires the united efforts o...</code> | <code>Toes connected at base.<br><br>RANGE--North America in general, breeding in the Arctic and sub-arctic districts, winters from<br>the Gulf States to Brazil.<br><br>NEST--Depression in the ground, with lining of dry grass.<br><br>EGGS--Three or four; buffy white, spotted with chocolate.<br></code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | <code>0</code> |
  | <code>The king instantly became hysterical with laughter. <br><br>"Trueth!" he said and turned to Kate, "thou art the shrewdest maiden in my kingdom." Turning to the baron he said: "This lady spoke the truth, and the goose shall only lay golden eggs." And thusly was Baron Von Dunderhead and his case dismissed.<br></code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | <code>I expect you're awfully rich. I'm poorer than any church<br>mouse. It doesn't look as if I could do anything for one like you. But who knows? There was<br>a mouse once helped a lion. It gnawed a hole in a net. I feel as if the time must come when<br>I can do as much, because I want to so dreadfully. That's all!"<br><br>IV<br><br>THE MURMUR OF THE STORM<br><br>It seemed that everything were to go wrong with Roger Sands that day. He had felt for the last<br>few months that a cloud had risen between him and John Heron, whose cause he had won in California.<br>If ever a business man owed a debt of gratitude to the brains of another, John Heron owed such<br>a debt to Roger Sands, who had risked not only his reputation, but even his life against the<br>powerful enemies of the alleged "California Oil Trust King." Heron had appeared fully to appreciate<br>this; and before Roger left for New York had been almost oppressively cordial, begging in vain<br>that Roger would visit him and his wife, a famous beauty with Spanish blood in her vein...</code> | <code>0</code> |
* Loss: <code>__main__.SEF1MSE</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 4
- `max_grad_norm`: 2.0
- `num_train_epochs`: 1
- `lr_scheduler_type`: constant
- `fp16`: True
- `optim`: sgd
- `dataloader_pin_memory`: False

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 4
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 2.0
- `num_train_epochs`: 1
- `max_steps`: -1
- `lr_scheduler_type`: constant
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: sgd
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: False
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch | Step | Training Loss |
|:-----:|:----:|:-------------:|
| 1.0   | 1    | 0.0602        |


### Framework Versions
- Python: 3.12.9
- Sentence Transformers: 5.1.2
- Transformers: 4.56.1
- PyTorch: 2.9.1+cu126
- Accelerate: 1.12.0
- Datasets: 4.4.1
- Tokenizers: 0.22.0

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->