from utility import *

test_dataset = load_file(test_dataset_path, "pickle")
trdc_device = torch.device("cuda")
trdc_model = build_trdc(trdc_device)[0]
model_checkpoint = torch.load(model_checkpoint_path, trdc_device)
trdc_model.load_state_dict(model_checkpoint["model"])

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    per_device_batch_size,
    num_workers=per_device_worker_count,
    collate_fn=DatasetBatch.load_batch,
    pin_memory=True
)

print("execute: running the model")
assess_trdc(trdc_device, trdc_model, test_loader)
