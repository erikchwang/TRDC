from utility import *

torch.distributed.init_process_group("nccl")
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
train_dataset = load_file(train_dataset_path, "pickle")
develop_dataset = load_file(develop_dataset_path, "pickle")
trdc_device = torch.device("cuda")
trdc_model, trdc_optimizer, trdc_scheduler = build_trdc(trdc_device)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    per_device_batch_size,
    sampler=torch.utils.data.distributed.DistributedSampler(train_dataset),
    num_workers=per_device_worker_count,
    collate_fn=DatasetBatch.load_batch,
    pin_memory=True
)

develop_loader = torch.utils.data.DataLoader(
    develop_dataset,
    per_device_batch_size,
    num_workers=per_device_worker_count,
    collate_fn=DatasetBatch.load_batch,
    pin_memory=True
)

while True:
    torch.distributed.barrier()
    begin_time = datetime.datetime.now()
    model_checkpoint = torch.load(model_checkpoint_path, trdc_device)

    if model_checkpoint["round"] > early_stopping_round_limit:
        break

    trdc_model.load_state_dict(model_checkpoint["model"])
    trdc_optimizer.load_state_dict(model_checkpoint["optimizer"])
    trdc_scheduler.load_state_dict(model_checkpoint["scheduler"])
    train_loader.sampler.set_epoch(trdc_scheduler.last_epoch)
    trdc_model = ParallelWrapper(trdc_model, [trdc_device], trdc_device, find_unused_parameters=True)
    update_trdc(trdc_device, trdc_model, trdc_optimizer, train_loader)
    trdc_model = trdc_model.module

    if torch.distributed.get_rank() == 0:
        overall_f1 = assess_trdc(trdc_device, trdc_model, develop_loader)

        if trdc_scheduler.is_better(overall_f1, trdc_scheduler.best):
            trdc_result = True
            model_checkpoint["model"] = trdc_model.state_dict()

        else:
            trdc_result = False
            model_checkpoint["round"] += 1
            trdc_optimizer.load_state_dict(model_checkpoint["optimizer"])

        trdc_scheduler.step(overall_f1)
        model_checkpoint["optimizer"] = trdc_optimizer.state_dict()
        model_checkpoint["scheduler"] = trdc_scheduler.state_dict()
        torch.save(model_checkpoint, model_checkpoint_path)

        print(
            "optimize: cost {} seconds in epoch {} and {}".format(
                int((datetime.datetime.now() - begin_time).total_seconds()),
                trdc_scheduler.last_epoch,
                "save the model" if trdc_result else "restore the model to the last saved epoch"
            )
        )
