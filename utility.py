import argparse, datetime, functools, glob, itertools, json, logging, multiprocessing
import os, pickle, random, torch, transformers, warnings

logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

transformers_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "transformers")
task_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "task")
posture_vocabulary_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset", "posture_vocabulary")
train_dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset", "train_dataset")
develop_dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset", "develop_dataset")
test_dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset", "test_dataset")
model_checkpoint_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoint", "model_checkpoint")
token_array_maximum_size = 512
learning_rate_initial_value = 0.00001
learning_rate_decay_rate = 0.5
posture_probability_threshold_value = 0.5
per_device_batch_size = 8
per_device_worker_count = 2
early_stopping_round_limit = 3


def load_file(file_path, file_type):
    if file_type == "pickle":
        with open(file_path, "rb") as stream:
            return pickle.load(stream)

    elif file_type == "text":
        with open(file_path, "rt") as stream:
            return stream.read().splitlines()

    else:
        raise Exception("invalid file type: {}".format(file_type))


def dump_file(file_items, file_path, file_type):
    if file_type == "pickle":
        with open(file_path, "wb") as stream:
            pickle.dump(file_items, stream)

    elif file_type == "text":
        with open(file_path, "wt") as stream:
            stream.write("\n".join(file_items))

    else:
        raise Exception("invalid file type: {}".format(file_type))


def convert_dataset(dataset_document, wordpiece_tokenizer, posture_vocabulary):
    parsed_document = json.loads(dataset_document)

    token_array = wordpiece_tokenizer.encode(
        " ".join(itertools.chain.from_iterable(section["paragraphs"] for section in parsed_document["sections"])),
        truncation=True,
        max_length=token_array_maximum_size
    )

    label_array = list(1 if posture in parsed_document["postures"] else 0 for posture in posture_vocabulary)
    dataset_example = {"token_array": token_array, "label_array": label_array}

    return dataset_example


class DatasetBatch:
    def __init__(self, token_arrays, token_counts, label_arrays):
        self.token_arrays = token_arrays
        self.token_counts = token_counts
        self.label_arrays = label_arrays

    @classmethod
    def load_dataset(cls, dataset_examples):
        token_counts = list(len(example["token_array"]) for example in dataset_examples)
        maximum_count = max(token_counts)

        token_arrays = torch.stack(
            list(
                torch.cat(
                    [
                        torch.tensor(example["token_array"], dtype=torch.long),
                        torch.zeros([maximum_count - count], dtype=torch.long)
                    ]
                )
                for example, count in zip(dataset_examples, token_counts)
            )
        )

        token_counts = torch.tensor(token_counts, dtype=torch.long)
        label_arrays = torch.tensor(list(example["label_array"] for example in dataset_examples), dtype=torch.long)

        return cls(token_arrays, token_counts, label_arrays)

    def pin_memory(self):
        self.token_arrays = self.token_arrays.pin_memory()
        self.token_counts = self.token_counts.pin_memory()
        self.label_arrays = self.label_arrays.pin_memory()

        return self

    def to(self, *args, **kwargs):
        self.token_arrays = self.token_arrays.to(*args, **kwargs)
        self.token_counts = self.token_counts.to(*args, **kwargs)
        self.label_arrays = self.label_arrays.to(*args, **kwargs)

        return self


class TRDCModel(torch.nn.Module):
    def __init__(self, context_encoder, posture_predictor):
        super().__init__()
        self.context_encoder = context_encoder
        self.posture_predictor = posture_predictor

    def forward(self, token_arrays, token_counts):
        prediction_arrays = self.posture_predictor(
            self.context_encoder(
                token_arrays,
                torch.lt(
                    torch.unsqueeze(
                        torch.arange(token_arrays.size()[1], dtype=torch.long, device=token_arrays.device),
                        0
                    ),
                    torch.unsqueeze(token_counts, 1)
                ).float()
            )[1]
        )

        return prediction_arrays


class ParallelWrapper(torch.nn.parallel.DistributedDataParallel):
    def __getattr__(self, attribute_name):
        try:
            return super().__getattr__(attribute_name)

        except:
            return getattr(self.module, attribute_name)


def build_trdc(trdc_device):
    posture_vocabulary = load_file(posture_vocabulary_path, "text")
    model_config = transformers.AutoConfig.from_pretrained(transformers_path)
    wordpiece_tokenizer = transformers.AutoTokenizer.from_pretrained(transformers_path)
    context_encoder = transformers.AutoModel.from_pretrained(transformers_path)
    posture_predictor = torch.nn.Linear(model_config.hidden_size, len(posture_vocabulary))

    with torch.no_grad():
        posture_predictor.weight.copy_(
            context_encoder(
                **wordpiece_tokenizer(
                    posture_vocabulary,
                    padding=True,
                    return_tensors="pt"
                )
            )[1]
        )

    trdc_model = TRDCModel(context_encoder, posture_predictor)
    trdc_model.to(trdc_device)

    trdc_optimizer = torch.optim.AdamW(
        [
            {
                "params": list(
                    parameter
                    for name, parameter in trdc_model.named_parameters()
                    if all(term not in name.lower() for term in ["bias", "norm"])
                )
            },
            {
                "params": list(
                    parameter
                    for name, parameter in trdc_model.named_parameters()
                    if any(term in name.lower() for term in ["bias", "norm"])
                ),
                "weight_decay": 0.0
            }
        ],
        learning_rate_initial_value
    )

    trdc_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trdc_optimizer, "max", learning_rate_decay_rate, 0)

    return trdc_model, trdc_optimizer, trdc_scheduler


def update_trdc(trdc_device, trdc_model, trdc_optimizer, dataset_loader):
    loss_criterion = torch.nn.BCEWithLogitsLoss()
    trdc_model.train()

    for dataset_batch in dataset_loader:
        dataset_batch.to(trdc_device, non_blocking=True)
        token_arrays = dataset_batch.token_arrays
        token_counts = dataset_batch.token_counts
        label_arrays = dataset_batch.label_arrays
        prediction_arrays = trdc_model(token_arrays, token_counts)
        loss_value = loss_criterion(prediction_arrays, label_arrays.float())
        trdc_optimizer.zero_grad()
        loss_value.backward()
        trdc_optimizer.step()


def assess_trdc(trdc_device, trdc_model, dataset_loader):
    trdc_model.eval()
    true_positives = []
    false_positives = []
    false_negatives = []

    for dataset_batch in dataset_loader:
        with torch.no_grad():
            dataset_batch.to(trdc_device, non_blocking=True)
            token_arrays = dataset_batch.token_arrays
            token_counts = dataset_batch.token_counts
            label_arrays = dataset_batch.label_arrays
            prediction_arrays = trdc_model(token_arrays, token_counts)

            selection_arrays = torch.gt(
                torch.nn.functional.sigmoid(prediction_arrays),
                torch.tensor(posture_probability_threshold_value, dtype=torch.float, device=prediction_arrays.device)
            ).long()

            true_positives.append(torch.sum(torch.mul(label_arrays, selection_arrays)).tolist())
            false_positives.append(torch.sum(selection_arrays).tolist() - true_positives[-1])
            false_negatives.append(torch.sum(label_arrays).tolist() - true_positives[-1])

    true_positive = sum(true_positives)
    false_positive = sum(false_positives)
    false_negative = sum(false_negatives)

    if true_positive == 0:
        f1_score = 0.0

    else:
        precision_score = true_positive / (true_positive + false_positive)
        recall_score = true_positive / (true_positive + false_negative)
        f1_score = 2.0 * precision_score * recall_score / (precision_score + recall_score)

    print("f1 score: {}".format(f1_score))

    return f1_score
