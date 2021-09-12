from utility import *

begin_time = datetime.datetime.now()
trdc_device = torch.device("cuda")
trdc_model, trdc_optimizer, trdc_scheduler = build_trdc(trdc_device)

model_checkpoint = {
    "round": 0,
    "model": trdc_model.state_dict(),
    "optimizer": trdc_optimizer.state_dict(),
    "scheduler": trdc_scheduler.state_dict()
}

torch.save(model_checkpoint, model_checkpoint_path)
print("construct: cost {} seconds".format(int((datetime.datetime.now() - begin_time).total_seconds())))
