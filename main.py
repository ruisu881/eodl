from eodl_dataloader import get_dataset
import torch.nn.functional as F
from torch.optim import SGD
from eodl_args import arg_parser
import torch
import models
from tqdm import tqdm
import json
import os

# Refresh frequency of tqdm
REFRESH_FREQUENCY = 10

def save_results(data,suffix = None):
    
    root_path = data['config']['root']
    dataset =  data['config']['dataset'] 
    modelname = data['config']['model']
    if suffix:
        modelname = modelname + suffix
    filename = data['config']['filename']

    path = os.path.join(root_path,dataset,modelname)

    if not os.path.exists(path):
        os.makedirs(path)
    
    with open(os.path.join(path,filename),'w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False)

def get_pred_label(pred,weights):
    emsemble_pred = torch.sum(weights.unsqueeze(1)*pred,dim=0).unsqueeze(0)
    return emsemble_pred.argmax().item()

def torch_train(optimizer,pred,label,weights = None, del_da = False):
    
    optimizer.zero_grad(set_to_none=True)
    
    pred_ = pred.detach()
    losses = torch.stack([F.cross_entropy(p.unsqueeze(0),label) for p in pred_])
    # labeles = label.expand(pred.shape[0])
    # losses = F.cross_entropy(pred,labeles,reduction='none')

    update_pred = torch.sum(weights.unsqueeze(1)*pred,dim=0).unsqueeze(0)
    loss = F.cross_entropy(update_pred,label)
    loss.backward()

    optimizer.step()
    return loss.item(),losses
    # return loss.item()

def main():
    
    eodl_args = arg_parser.parse_args()
    print(eodl_args)

    try:
        dataloader,dataset = get_dataset(eodl_args)
        eodl_args.nIn = dataset.get_nIn()
        eodl_args.nOut = dataset.get_nOut()
        eodl_args.total = dataset.__len__()

    except FileNotFoundError:
        print('dataset not support!')


    model = getattr(models,eodl_args.model)(eodl_args)

    optimizer = SGD(model.parameters(),lr = eodl_args.lr)
    print(model)


    pbar = tqdm(total=eodl_args.total,ncols=100) 
    

    data_output = {}
    data_output['config'] = eodl_args.__dict__
    data_output['pred'] = []
    data_output['label'] = []
    data_output['alphas'] = []
    data_output['gamma'] = []

    for i,(x,y) in enumerate(dataloader):
        pred = model(x)
        
        loss,losses = torch_train(optimizer,pred,y,weights=model.gamma_vector)

        pl = get_pred_label(pred,model.alpha_vector)
        
        model.update_weights(losses)

        data_output['pred'].append(pl)
        data_output['label'].append(y.item())
        data_output['alphas'].append(model.alpha_vector.tolist())
        data_output['gamma'].append(model.gamma_vector.tolist())

        if i % REFRESH_FREQUENCY == 0:
            pbar.set_description('Loss:{:.4f}'.format(loss))
            pbar.update(REFRESH_FREQUENCY)

    if eodl_args.del_all:
        suffix = "_DEL_ALL"
    elif eodl_args.del_da:
        suffix = "_DEL_DA"
    elif eodl_args.del_pa:
        suffix = "_DEL_PA"
    save_results(data_output,suffix)

if __name__ == '__main__':

    main()
    

