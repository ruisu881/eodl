import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class EODL(nn.Module):

    def __init__(self, args):
        super(EODL, self).__init__()
        self.config = args

        self.init_layers()      

    def init_layers(self):
        # intializa network, alpha vector and gamma vector

        hl_num = self.config.ln - 1
        hidden_layers = []
        hidden_layers.append(nn.Linear(self.config.nIn,self.config.nHn))
        hidden_layers.extend([nn.Linear(self.config.nHn, self.config.nHn) for _ in range(hl_num)])
        self.hidden_layers = nn.ModuleList(hidden_layers)

        output_layers = []
        output_layers = [nn.Linear(self.config.nHn, self.config.nOut) for _ in range(hl_num)]
        self.output_layers = nn.ModuleList(output_layers)

        # init pred weights alpha
        self.alpha_vector = torch.full(size=(hl_num,),fill_value=1./hl_num,requires_grad=False)
        # init update weights gamma
        self.gamma_vector = torch.full(size=(hl_num,),fill_value=1.,requires_grad=False)

        self.v = torch.full(size=(hl_num,),fill_value=1.,requires_grad=False)

    def concept_drift_detect(self):
        # detect whether concept drift occurs with eq. 5
         
        return torch.std(self.alpha_vector) <= self.config.theta

    def update_weights(self,losses):
        
        if not self.config.del_da and not self.config.del_all:
        # del_da don't need to update alpha

            # update decay vector of pred weights by v_(t-1), beta, loss_(t) and p with eq.9
            self.v= self.config.p * self.v + (1-self.config.p) * torch.pow(self.config.beta,losses)
            
            # compute alpha_t+1 with eq. 10 and normalization
            new_alpha = torch.clamp(self.alpha_vector * self.v, min=self.config.smooth / self.alpha_vector.shape[0])
            self.alpha_vector = torch.divide(new_alpha, torch.sum(new_alpha))

        # get the detection of concept drift
        result = self.concept_drift_detect()


        if result:
            #concept drift occurs

            # update gamma with eq. 11
            self.gamma_vector = torch.tensor([1.-torch.max(self.alpha_vector) for _ in range(self.alpha_vector.shape[0])])
        else:
            # concept drift isn't occurs
            self.lw = torch.argmax(self.alpha_vector)

            # compute the backpropagation depth for intermediate classifiers with eq. 7
            indexs = torch.tensor(np.arange(self.alpha_vector.shape[0]))
            l_best = torch.argmax(self.alpha_vector) # eq. 6
            split_index = l_best if l_best == self.alpha_vector.shape[0] else l_best + 1 
            distance = torch.abs(indexs - l_best)
            dis_decay = distance / (torch.max(distance) + 1e-5)

            # update gamma with eq. 11
            self.gamma_vector = torch.where(torch.lt(indexs,split_index), 1. - torch.max(self.alpha_vector), torch.exp(-(dis_decay)))

        if self.config.del_pa or self.config.del_all:
            # del_pa don't need to update gamma
            self.gamma_vector = self.alpha_vector.clone()

    def forward(self, x):
        
        outputs = []
        out = F.relu(self.hidden_layers[0](x))

        CD_result = self.concept_drift_detect()
        

        if self.config.del_da or self.config.del_all:
            # For del_da and del_all, we can simulate the forward propagation by setting CD_result to True

            CD_result = True

        # not CD_result indicates that concept drift did not occur;
        if not CD_result:
            l_best = torch.argmax(self.alpha_vector)
            split_index = l_best if l_best == self.alpha_vector.shape[0] else l_best + 1
            out_ = out

        for i,(hl,ol) in enumerate(zip(self.hidden_layers[1:],self.output_layers)):

            if not CD_result and i <= split_index:
                out_ = F.relu(hl(out_))
                
            out = F.relu(hl(out))

            if not CD_result and i in [l_best,split_index]:
                outputs.append(ol(out_))
            else:
                outputs.append(ol(out))

            if CD_result:
                out = out.detach()
            elif i > split_index or i < l_best:
                out = out.detach()


        return torch.concat(outputs,dim=0)
            

        
        