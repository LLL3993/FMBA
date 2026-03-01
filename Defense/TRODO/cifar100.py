import torch
from src.constants import NORM_MEAN, NORM_STD
from src.constants import num_classes as num_classes_dict
from src.models.loaders import load_resnet as model_loader
from src.modelset import ModelDataset

dataset = 'cifar100'

batch_size = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

num_classes = num_classes_dict[dataset]

final_model_loader = lambda x, meta_data: model_loader(x,
                                                       num_classes=num_classes,
                                                       mean=NORM_MEAN[dataset],
                                                       std=NORM_STD[dataset],
                                                       normalize=True,
                                                       meta_data=meta_data)

clean_root = '/data1/liuyiyang/safe-work/TRODO/model/clean'
trojaned_root = '/data1/liuyiyang/safe-work/TRODO/model/trojaned'

test_modelset = ModelDataset(clean_root,
                             trojaned_root,
                             final_model_loader
                             )

print("No. clean models in test set:", len([m for m in test_modelset.models_data if m['label'] == 0]))
print("No. trojaned models in test set:", len([m for m in test_modelset.models_data if m['label'] == 1]))

from src.data.loaders import get_near_ood_loader

def get_dataloader():
    dataloader = get_near_ood_loader(source_dataset=dataset, batch_size=batch_size)
    # print("Size of dataset:", len(dataloader.dataset))
    return dataloader

dataloader = get_dataloader()
print(len(dataloader.dataset))
# visualize_samples(dataloader, 1)


from src.evaluate import evaluate_modelset, mean_id_score_diff

evaluate_modelset(test_modelset,
                  signature_function=mean_id_score_diff,
                  signature_function_kwargs={
                    'eps': 2/255,
                    'device': device,
                    'verbose': True,
                  },
                  get_dataloader_func=get_dataloader,
                  progress=False,)


