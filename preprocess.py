from utility import *

begin_time = datetime.datetime.now()
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("step_index", type=int)

if argument_parser.parse_args().step_index == 0:
    document_count = 0
    posture_count = 0
    paragraph_count = 0
    posture_frequency = {}

    for path in glob.glob("{}/*".format(task_path)):
        dataset_documents = load_file(path, "text")
        document_count += len(dataset_documents)

        for document in dataset_documents:
            parsed_document = json.loads(document)
            posture_count += len(parsed_document["postures"])
            paragraph_count += sum(len(section["paragraphs"]) for section in parsed_document["sections"])

            for posture in parsed_document["postures"]:
                if posture in posture_frequency:
                    posture_frequency[posture] += 1

                else:
                    posture_frequency[posture] = 1

    posture_vocabulary = []
    posture_weight = []

    for posture in sorted(posture_frequency, key=posture_frequency.get, reverse=True):
        posture_vocabulary.append(posture)
        posture_weight.append((document_count - posture_frequency[posture]) / posture_frequency[posture])

    dump_file(posture_vocabulary, posture_vocabulary_path, "text")
    dump_file(posture_weight, posture_weight_path, "pickle")

    print(
        "document count: {}, posture count: {} (distinct: {}), paragraph count: {}".format(
            document_count,
            posture_count,
            len(posture_frequency),
            paragraph_count
        )
    )

elif argument_parser.parse_args().step_index == 1:
    with multiprocessing.Pool() as pool:
        posture_vocabulary = load_file(posture_vocabulary_path, "text")
        wordpiece_tokenizer = transformers.AutoTokenizer.from_pretrained(transformers_path)

        dataset_examples = pool.map(
            functools.partial(
                convert_dataset,
                wordpiece_tokenizer=wordpiece_tokenizer,
                posture_vocabulary=posture_vocabulary
            ),
            itertools.chain.from_iterable(load_file(path, "text") for path in glob.glob("{}/*".format(task_path)))
        )

        random.shuffle(dataset_examples)
        train_split = int(len(dataset_examples) * 0.8)
        train_dataset = dataset_examples[:train_split]
        develop_split = int((len(dataset_examples) - train_split) * 0.5)
        develop_dataset = dataset_examples[train_split:][:develop_split]
        test_dataset = dataset_examples[train_split:][develop_split:]
        dump_file(train_dataset, train_dataset_path, "pickle")
        dump_file(develop_dataset, develop_dataset_path, "pickle")
        dump_file(test_dataset, test_dataset_path, "pickle")

        print(
            "train dataset size: {}, develop dataset size: {}, test dataset size: {}".format(
                len(train_dataset),
                len(develop_dataset),
                len(test_dataset)
            )
        )

else:
    raise Exception("invalid step index: {}".format(argument_parser.parse_args().step_index))

print(
    "preprocess step {}: cost {} seconds".format(
        argument_parser.parse_args().step_index,
        int((datetime.datetime.now() - begin_time).total_seconds())
    )
)
