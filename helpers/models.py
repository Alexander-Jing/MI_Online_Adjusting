import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange, repeat
from torch import nn, einsum
from einops.layers.torch import Rearrange
import math
from dn3.trainable.layers import Flatten


#hz confirmed with keras implemenation of the official EEGNet: https://github.com/vlawhern/arl-eegmodels/blob/f3218571d1e7cfa828da0c697299467ea101fd39/EEGModels.py#L359


#assume using window size 200ts
#feature_size = 8
#timestep = 200
#F1 = 16
#D = 2
#F2 = D * F1
#output of each layer see HCI/NuripsDataSet2021/ExploreEEGNet_StepByStep.ipynb

#Conv2d with Constraint (https://github.com/braindecode/braindecode/blob/master/braindecode/models/eegnet.py)
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
        
    def forward(self, x):
        # this function is used to normalize the convolution kernel in the dim 0 (batch dimension), making the p-norm value in each normalize batch less than maxnorm
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        
        return super(Conv2dWithConstraint, self).forward(x)

class EEGNet150(nn.Module):
    def __init__(self, feature_size=8, num_timesteps=150, num_classes=2, F1=4, D=2, F2=8, dropout=0.5):

        super(EEGNet150, self).__init__()

        #Temporal convolution
        self.firstConv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False), 
            #'same' padding: used by the author; 
            #kernel_size=(1,3):  author recommend kernel length be half of the sampling rate
            nn.BatchNorm2d(num_features=F1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.depthwiseConv = nn.Sequential(
            Conv2dWithConstraint(F1, F1 * D, kernel_size=(feature_size, 1), stride=(1, 1), groups=F1, bias=False),   # group convolution is used for the goal of fewer parameters
            #'valid' padding: used by the author;
            #kernel_size = (feature_size, 1): used by the author
            
            nn.BatchNorm2d(F1 * D, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(), #used by author
            nn.AvgPool2d(kernel_size=(1, 4)), #kernel_size=(1,4) used by author
            nn.Dropout(p=dropout) 
        )

        #depthwise convolution follow by pointwise convolution (pointwise convolution is just Conv2d with 1x1 kernel)
        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 3), stride=1, padding=(0, 1), groups=F1*D, bias=False),
            nn.Conv2d(F2, F2, kernel_size=1, bias=False),
            nn.BatchNorm2d(F2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(),
            nn.ELU(), #use by author
            nn.AvgPool2d(kernel_size=(1, 8)), #kernel_size=(1,8): used by author
            nn.Dropout(p=dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=32, out_features=num_classes, bias=True)
        ) #先不implement最后一层的kernel constraint， 只implement conv2d的constraint

    def forward(self, x):
        x = self.firstConv(x.unsqueeze(1).transpose(2,3))  # (2,3)转置将150*8转置成8*150
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        # print(x.shape)
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x = self.classifier(x)
        normalized_probabilities= F.log_softmax(x, dim = 1)     

        return normalized_probabilities #for EEGNet and DeepConvNet, directly use nn.NLLLoss() as criterion

# We used the EEGNet-8,2 as the model for online updating experiment, where F1=8, D=2 and F2=D*F1
# The kernel size is set as (1, 32)  according to the methods in:  
#    Ma X, Qiu S, Wei W, et al. Deep channel-correlation network for motor imagery decoding from the same limb[J]. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2019, 28(1): 297-306.
# Conv2d with Constraint is used(https://github.com/braindecode/braindecode/blob/master/braindecode/models/eegnet.py)
class EEGNetFea(nn.Module):
    def __init__(self, feature_size=30, num_timesteps=512, num_classes=3, F1=8, D=2, F2=16, dropout=0.5):

        super(EEGNetFea, self).__init__()

        #Temporal convolution
        self.firstConv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(1, 32), stride=1, padding=(0, 15), bias=False), 
            #'same' padding: used by the author; 
            #kernel_size=(1,32):  author recommend kernel length be half of the sampling rate
            nn.BatchNorm2d(num_features=F1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.depthwiseConv = nn.Sequential(
            Conv2dWithConstraint(F1, F1 * D, kernel_size=(feature_size, 1), stride=(1, 1), groups=F1, bias=False),   # group convolution is used for the goal of fewer parameters
            #'valid' padding: used by the author;
            #kernel_size = (feature_size, 1): used by the author
            
            nn.BatchNorm2d(F1 * D, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(), #used by author
            nn.AvgPool2d(kernel_size=(1, 4)), #kernel_size=(1,4) used by author
            nn.Dropout(p=dropout) 
        )

        #depthwise convolution follow by pointwise convolution (pointwise convolution is just Conv2d with 1x1 kernel)
        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 16), stride=1, padding=(0, 8), groups=F1*D, bias=False),
            nn.Conv2d(F2, F2, kernel_size=1, bias=False),
            nn.BatchNorm2d(F2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#             nn.ReLU(),
            nn.ELU(), #use by author
            nn.AvgPool2d(kernel_size=(1, 8)), #kernel_size=(1,8): used by author
            nn.Dropout(p=dropout)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=int(num_timesteps/32*F2), out_features=num_classes, bias=True)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.firstConv(x.float())  # unsqueeze the x dim
        x = self.depthwiseConv(x)
        features = self.separableConv(x)
        _features = features.squeeze(2)
        logits = self.classifier(features)    

        return logits, _features # use the cross-entrophy loss


#the author used kernel_size=(1,3) stride=(1,3) for all the MaxPool2d layer. Here we use less agressive down-sampling, because our input chunk has only 200 timesteps

#we didn't implement the tied-loss as described by the author, because our goal is to predict each chunk, while the goal of the paper is to predict each trial from all the chunks of this trial.
    
class DeepConvNet150(nn.Module):
    def __init__(self, feature_size=8, num_timesteps=150, num_classes=2, dropout=0.5):
        super(DeepConvNet150, self).__init__()
        
    
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1,5), stride=1, padding=0, bias=True),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(feature_size, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(), #use by author
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        self.block2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1,5), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(), #use by author
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        self.block3 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1, 5), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(), #use by author
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        self.block4 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(1, 5), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(), #use by author
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=200, out_channels=num_classes, kernel_size=(1, 5), bias=True)
        )

        
    def forward(self, x):
        x = self.block1(x.unsqueeze(1).transpose(2,3))
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        x = x.squeeze(dim=2).squeeze(dim=2)
        print(x.shape)
        normalized_probabilities = F.log_softmax(x, dim = 1)     

        return normalized_probabilities #for EEGNet and DeepConvNet, directly use nn.NLLLoss() as criterion

# We used the DeepConvNet for experiment
# Referring to the work: 
#    Schirrmeister R T, Springenberg J T, Fiederer L D J, et al. Deep learning with convolutional neural networks for EEG decoding and visualization[J]. Human brain mapping, 2017, 38(11): 5391-5420.
#    Ma X, Qiu S, Wei W, et al. Deep channel-correlation network for motor imagery decoding from the same limb[J]. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2019, 28(1): 297-306.
# Referring to the code in: 
#    https://github.com/TNTLFreiburg/braindecode/blob/master/braindecode/models/shallow_fbcsp.py
#    https://github.com/tufts-ml/fNIRS-mental-workload-classifiers

class DeepConvNetFea(nn.Module):
    def __init__(self, feature_size=30, num_timesteps=512, num_classes=3, dropout=0.5):
        super(DeepConvNetFea, self).__init__()
        
    
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1,10), stride=1, padding=0, bias=True),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(feature_size, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(), #use by author
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
        )

        self.block2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1,10), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(), #use by author
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
        )

        self.block3 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1, 10), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(), #use by author
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
        )

        self.block4 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(1, 10), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(), #use by author
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
        )

        self.classifier = nn.Sequential(
            #nn.Flatten(),
            #nn.Linear(in_features=200, out_features=num_classes, bias=True)
            nn.Conv2d(in_channels=200, out_channels=num_classes, kernel_size=(1, 1), bias=True)
        )
        
        
    def forward(self, x):
        x = self.block1(x.unsqueeze(1))
        x = self.block2(x)
        x = self.block3(x)
        features = self.block4(x)
        _features = features.squeeze(2)
        logits = self.classifier(features)
        logits = logits.squeeze(dim=2).squeeze(dim=2)

        return logits, _features  # we only use the cross-entrophy loss

# We used the DeepConvNet for experiment
# Referring to the work: 
#    Schirrmeister R T, Springenberg J T, Fiederer L D J, et al. Deep learning with convolutional neural networks for EEG decoding and visualization[J]. Human brain mapping, 2017, 38(11): 5391-5420.
#    Ma X, Qiu S, Wei W, et al. Deep channel-correlation network for motor imagery decoding from the same limb[J]. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2019, 28(1): 297-306.
# Referring to the code in: 
#    https://github.com/TNTLFreiburg/braindecode/blob/master/braindecode/models/shallow_fbcsp.py
#    https://github.com/tufts-ml/fNIRS-mental-workload-classifiers

class DeepConvNetFeaDence(nn.Module):
    def __init__(self, feature_size=30, num_timesteps=512, num_classes=3, dropout=0.5):
        super(DeepConvNetFeaDence, self).__init__()
        
    
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1,3), stride=1, padding=0, bias=True),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(feature_size, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(), #use by author
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
        )

        self.block2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1,3), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(), #use by author
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
        )

        self.block3 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1, 3), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(), #use by author
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
        )

        self.block4 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(1, 3), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(), #use by author
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=200*5, out_features=num_classes, bias=True)
            #nn.Conv2d(in_channels=200, out_channels=num_classes, kernel_size=(1, 1), bias=True)
        )
        
        
    def forward(self, x):
        x = self.block1(x.unsqueeze(1))
        x = self.block2(x)
        x = self.block3(x)
        features = self.block4(x)
        _features = features.squeeze(2)
        logits = self.classifier(features)
        #logits = logits.squeeze(dim=2).squeeze(dim=2)

        return logits, _features  # we only use the cross-entrophy loss

class ResBlockBN(nn.Module):
    def __init__(self, in_features, encoder_h, res_width=(3,3), res_stride=(1,1), dropout=0.):
        super(ResBlockBN, self).__init__()
        
        self.ResBlock_0 = nn.Sequential(
            nn.Conv1d(in_features, encoder_h, kernel_size=res_width[0], stride=res_stride[0], padding=res_width[0] // 2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(encoder_h),
            nn.GELU(),)
        
        self.ResBlock_1 = nn.Sequential()
        for i, (width, stride) in enumerate(zip(res_width, res_stride)):
            self.ResBlock_1.add_module("ResBlcok_Conv{}".format(i), nn.Sequential(
                nn.Conv1d(in_features, encoder_h, kernel_size=width, stride=stride, padding=width // 2),
                nn.Dropout(dropout),
                nn.BatchNorm1d(encoder_h),
                nn.GELU(),
            ))
            in_features = encoder_h
    def forward(self, x):
        return x + self.ResBlock_1(x)


class _BENDREncoder(nn.Module):
    def __init__(self, in_features, encoder_h=256,):
        super().__init__()
        self.in_features = in_features
        self.encoder_h = encoder_h

    def load(self, filename, strict=True):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict, strict=strict)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def freeze_features(self, unfreeze=False):
        for param in self.parameters():
            param.requires_grad = unfreeze


class ConvEncoderResBN(_BENDREncoder):
    def __init__(self, in_features, encoder_h=512, output_h=128, enc_width=((3,3), (3,3), (3,3), (3,3), (3,3)),
                 dropout=0., projection_head=False, enc_downsample=((1,1), (1,1), (1,1), (1,1), (1,1))):
        super().__init__(in_features, encoder_h)
        
        self.output_h = output_h
        if not isinstance(enc_width, (list, tuple)):
            enc_width = [enc_width]
        if not isinstance(enc_downsample, (list, tuple)):
            enc_downsample = [enc_downsample]
        assert len(enc_downsample) == len(enc_width)

        # Centerable resdiual convolutional neutral network based model 
        self._downsampling = enc_downsample
        self._width = enc_width

        self.encoder = nn.Sequential()
        self.encoder.add_module("Encoder_Input", nn.Sequential(
            nn.Conv1d(in_features, encoder_h, kernel_size=enc_width[0][0], stride=enc_width[0][0], padding=enc_width[0][0] // 2),
            nn.Dropout(dropout),
            nn.Conv1d(encoder_h, encoder_h, kernel_size=(enc_width[0][0]-1), stride=(enc_width[0][0]-1), padding=(enc_width[0][0]-1)// 2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(encoder_h),
            nn.GELU(),
        ))
        in_features = encoder_h

        for i, (width, downsample) in enumerate(zip(enc_width, enc_downsample)):
            self.encoder.add_module("Encoder_ResBlock{}".format(i), nn.Sequential(
                ResBlockBN(in_features, encoder_h, res_width=width, res_stride=downsample)
            ))
            in_features = encoder_h

        self.encoder.add_module("Encoder_Output", nn.Sequential(
            nn.Conv1d(encoder_h, output_h*2, kernel_size=enc_width[-1][-1], stride=enc_downsample[-1][-1], padding=enc_width[-1][-1] // 2),
            nn.Dropout(dropout),
            nn.Conv1d(output_h*2, output_h, kernel_size=enc_width[-1][-1], stride=enc_downsample[-1][-1], padding=enc_width[-1][-1] // 2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(output_h),
            nn.GELU(),
        ))

        if projection_head:
            self.encoder.add_module("projection-1", nn.Sequential(
                nn.Conv1d(in_features, in_features, 1),
                nn.Dropout(dropout*2),
                nn.BatchNorm1d(encoder_h),
                nn.GELU()
            ))

    def forward(self, x):
          
        return self.encoder(x)
    


class ConvEncoderCls(_BENDREncoder):
    def __init__(self, in_features=128, encoder_h=128, output_h=128, width=((3,3),), stride=((1,1),), dropout=0.0, targets=3, 
                 num_features_for_classification=int(15*64), multi_gpu=False, encoder_grad_frac=1.0):
        super().__init__(in_features, encoder_h)
        self.output_h = output_h
        if not isinstance(width, (list, tuple)):
            width = [width]
        if not isinstance(stride, (list, tuple)):
            stride = [stride]
        assert len(stride) == len(width)
        self.targets = targets
        self.num_features_for_classification = num_features_for_classification
        self.encoder = nn.Sequential()

        for i, (_width, downsample) in enumerate(zip(width, stride)):
            self.encoder.add_module("Encoder_ResBlock{}".format(i), nn.Sequential(
                ResBlockBN(output_h, output_h, res_width=_width, res_stride=downsample)
            ))
        
        self.encoder.add_module("Encoder_Output", nn.Sequential(
            nn.Conv1d(output_h, output_h, kernel_size=3, stride=3, padding=3 // 2),
            nn.Dropout(dropout),
            nn.Conv1d(output_h, int(output_h/2), kernel_size=2, stride=2, padding=2 // 2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(int(output_h/2)),
            nn.GELU(),
        ))
        self.encoder.add_module("Encoder_Cls", nn.Sequential(Flatten(), 
                                 nn.Linear(self.num_features_for_classification, self.targets))) 
    
    def forward(self, x):
        x = self.encoder.Encoder_ResBlock0(x)
        x = self.encoder.Encoder_Output(x)
        x = self.encoder.Encoder_Cls(x)

        return x
    
class ConvEncoderClsFea(_BENDREncoder):
    def __init__(self, in_features=128, encoder_h=128, output_h=128, width=((3,3),), stride=((1,1),), dropout=0.0, targets=3, 
                 num_features_for_classification=int(15*64), multi_gpu=False, encoder_grad_frac=1.0):
        super().__init__(in_features, encoder_h)
        self.output_h = output_h
        if not isinstance(width, (list, tuple)):
            width = [width]
        if not isinstance(stride, (list, tuple)):
            stride = [stride]
        assert len(stride) == len(width)
        self.targets = targets
        self.num_features_for_classification = num_features_for_classification
        self.encoder = nn.Sequential()
 
        for i, (_width, downsample) in enumerate(zip(width, stride)):
            self.encoder.add_module("Encoder_ResBlock{}".format(i), nn.Sequential(
                ResBlockBN(output_h, output_h, res_width=_width, res_stride=downsample)
            ))
        
        self.encoder.add_module("Encoder_Output", nn.Sequential(
            nn.Conv1d(output_h, output_h, kernel_size=3, stride=3, padding=3 // 2),
            nn.Dropout(dropout),
            nn.Conv1d(output_h, int(output_h/2), kernel_size=2, stride=2, padding=2 // 2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(int(output_h/2)),
            nn.GELU(),
        ))
        self.encoder.add_module("Encoder_Cls", nn.Sequential(Flatten(), 
                                 nn.Linear(self.num_features_for_classification, self.targets))) 
    
    def forward(self, x):
        x = self.encoder.Encoder_ResBlock0(x)
        features = self.encoder.Encoder_Output(x)
        logits = self.encoder.Encoder_Cls(features)

        return logits, features

class ConvEncoder_ClsFea(_BENDREncoder):
    def __init__(self, in_features=30, encoder_h=128, output_h=128, width=((3,3),), stride=((1,1),), dropout=0.0, targets=3, 
                 num_features_for_classification=int(15*64), multi_gpu=False, encoder_grad_frac=1.0):
        super().__init__(in_features, encoder_h)
        self.output_h = output_h
        if not isinstance(width, (list, tuple)):
            width = [width]
        if not isinstance(stride, (list, tuple)):
            stride = [stride]
        assert len(stride) == len(width)
        self.targets = targets
        self.num_features_for_classification = num_features_for_classification
        self.encoder = nn.Sequential()

        self.encoder.add_module("Encoder_Input", nn.Sequential(
            nn.Conv1d(in_features, output_h, kernel_size=width[0][0], stride=width[0][0], padding=width[0][0] // 2),
            nn.Dropout(dropout),
            nn.Conv1d(output_h, output_h, kernel_size=(width[0][0]-1), stride=(width[0][0]-1), padding=(width[0][0]-1)// 2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(output_h),
            nn.GELU(),
        ))

        for i, (_width, downsample) in enumerate(zip(width, stride)):
            self.encoder.add_module("Encoder_ResBlock{}".format(i), nn.Sequential(
                ResBlockBN(output_h, output_h, res_width=_width, res_stride=downsample)
            ))
        
        self.encoder.add_module("Encoder_Output", nn.Sequential(
            nn.Conv1d(output_h, output_h, kernel_size=3, stride=3, padding=3 // 2),
            nn.Dropout(dropout),
            nn.Conv1d(output_h, int(output_h/2), kernel_size=2, stride=2, padding=2 // 2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(int(output_h/2)),
            nn.GELU(),
        ))
        self.encoder.add_module("Encoder_Cls", nn.Sequential(Flatten(), 
                                 nn.Linear(self.num_features_for_classification, self.targets))) 
    
    def forward(self, x):
        x = self.encoder.Encoder_Input(x)
        x = self.encoder.Encoder_ResBlock0(x)
        features = self.encoder.Encoder_Output(x)
        logits = self.encoder.Encoder_Cls(features)

        return logits


class ConvEncoder_ClsFeaTL(_BENDREncoder):
    def __init__(self, in_features=30, encoder_h=128, output_h=128, width=((3,3),), stride=((1,1),), dropout=0.0, targets=3, 
                 num_features_for_classification=int(15*64), multi_gpu=False, encoder_grad_frac=1.0):
        super().__init__(in_features, encoder_h)
        self.output_h = output_h
        if not isinstance(width, (list, tuple)):
            width = [width]
        if not isinstance(stride, (list, tuple)):
            stride = [stride]
        assert len(stride) == len(width)
        self.targets = targets
        self.num_features_for_classification = num_features_for_classification
        self.encoder = nn.Sequential()

        self.encoder.add_module("Encoder_Input", nn.Sequential(
            nn.Conv1d(in_features, output_h, kernel_size=width[0][0], stride=width[0][0], padding=width[0][0] // 2),
            nn.Dropout(dropout),
            nn.Conv1d(output_h, output_h, kernel_size=(width[0][0]-1), stride=(width[0][0]-1), padding=(width[0][0]-1)// 2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(output_h),
            nn.GELU(),
        ))

        for i, (_width, downsample) in enumerate(zip(width, stride)):
            self.encoder.add_module("Encoder_ResBlock{}".format(i), nn.Sequential(
                ResBlockBN(output_h, output_h, res_width=_width, res_stride=downsample)
            ))
        
        self.encoder.add_module("Encoder_Output", nn.Sequential(
            nn.Conv1d(output_h, output_h, kernel_size=3, stride=3, padding=3 // 2),
            nn.Dropout(dropout),
            nn.Conv1d(output_h, int(output_h/2), kernel_size=2, stride=2, padding=2 // 2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(int(output_h/2)),
            nn.GELU(),
        ))
        self.encoder.add_module("Encoder_Cls", nn.Sequential(Flatten(), 
                                 nn.Linear(self.num_features_for_classification, self.targets))) 
    
    def forward(self, x):
        x = self.encoder.Encoder_Input(x)
        x = self.encoder.Encoder_ResBlock0(x)
        features = self.encoder.Encoder_Output(x)
        logits = self.encoder.Encoder_Cls(features)

        return logits, features

class ConvEncoder3_ClsFeaTL(_BENDREncoder):
    def __init__(self, in_features=30, encoder_h=128, output_h=128, width=((3,3),(3,3),(3,3)), stride=((1,1),(1,1),(1,1)), dropout=0.0, targets=3, 
                 num_features_for_classification=int(15*64), multi_gpu=False, encoder_grad_frac=1.0):
        super().__init__(in_features, encoder_h)
        self.output_h = output_h
        if not isinstance(width, (list, tuple)):
            width = [width]
        if not isinstance(stride, (list, tuple)):
            stride = [stride]
        assert len(stride) == len(width)
        self.targets = targets
        self.num_features_for_classification = num_features_for_classification
        self.encoder = nn.Sequential()

        self.encoder.add_module("Encoder_Input", nn.Sequential(
            nn.Conv1d(in_features, output_h, kernel_size=width[0][0], stride=width[0][0], padding=width[0][0] // 2),
            nn.Dropout(dropout),
            nn.Conv1d(output_h, output_h, kernel_size=(width[0][0]-1), stride=(width[0][0]-1), padding=(width[0][0]-1)// 2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(output_h),
            nn.GELU(),
        ))

        for i, (_width, downsample) in enumerate(zip(width, stride)):
            self.encoder.add_module("Encoder_ResBlock{}".format(i), nn.Sequential(
                ResBlockBN(output_h, output_h, res_width=_width, res_stride=downsample)
            ))
        
        self.encoder.add_module("Encoder_Output", nn.Sequential(
            nn.Conv1d(output_h, output_h, kernel_size=3, stride=3, padding=3 // 2),
            nn.Dropout(dropout),
            nn.Conv1d(output_h, int(output_h/2), kernel_size=2, stride=2, padding=2 // 2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(int(output_h/2)),
            nn.GELU(),
        ))
        self.encoder.add_module("Encoder_Cls", nn.Sequential(Flatten(), 
                                 nn.Linear(self.num_features_for_classification, self.targets))) 
    
    def forward(self, x):
        x = self.encoder.Encoder_Input(x)
        x = self.encoder.Encoder_ResBlock0(x)
        features = self.encoder.Encoder_Output(x)
        logits = self.encoder.Encoder_Cls(features)

        return logits, features

class ConvEncoder3ResBN(_BENDREncoder):
    def __init__(self, in_features, encoder_h=256, output_h=128, enc_width=((3,3), (3,3), (3,3),),
                 dropout=0., projection_head=False, enc_downsample=((1,1), (1,1), (1,1),), use_output=False):
        super().__init__(in_features, encoder_h)
        self.encoder_h = encoder_h
        self.output_h = output_h
        if not isinstance(enc_width, (list, tuple)):
            enc_width = [enc_width]
        if not isinstance(enc_downsample, (list, tuple)):
            enc_downsample = [enc_downsample]
        assert len(enc_downsample) == len(enc_width)

        # Centerable resdiual convolutional neutral network based model 
        self._downsampling = enc_downsample
        self._width = enc_width

        self.encoder = nn.Sequential()
        self.encoder.add_module("Encoder_Input", nn.Sequential(
            nn.Conv1d(in_features, encoder_h, kernel_size=enc_width[0][0], stride=enc_width[0][0], padding=enc_width[0][0] // 2),
            nn.Dropout(dropout),
            nn.Conv1d(encoder_h, encoder_h, kernel_size=(enc_width[0][0]-1), stride=(enc_width[0][0]-1), padding=(enc_width[0][0]-1)// 2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(encoder_h),
            nn.GELU(),
        ))
        in_features = encoder_h

        for i, (width, downsample) in enumerate(zip(enc_width, enc_downsample)):
            self.encoder.add_module("Encoder_ResBlock{}".format(i), nn.Sequential(
                ResBlockBN(in_features, encoder_h, res_width=width, res_stride=downsample)
            ))
            in_features = encoder_h

        if use_output:
            self.encoder.add_module("Encoder_Output", nn.Sequential(
                nn.Conv1d(encoder_h, output_h, kernel_size=enc_width[-1][-1], stride=enc_downsample[-1][-1], padding=enc_width[-1][-1] // 2),
                nn.Dropout(dropout),
                nn.Conv1d(output_h, output_h, kernel_size=enc_width[-1][-1], stride=enc_downsample[-1][-1], padding=enc_width[-1][-1] // 2),
                nn.Dropout(dropout),
                nn.BatchNorm1d(output_h),
                nn.GELU(),
            ))

        if projection_head:
            self.encoder.add_module("projection-1", nn.Sequential(
                nn.Conv1d(in_features, in_features, 1),
                nn.Dropout(dropout*2),
                nn.BatchNorm1d(encoder_h),
                nn.GELU()
            ))

    def forward(self, x):
          
        return self.encoder(x)

class ConvEncoder_OutputClsFeaTL(_BENDREncoder):
    def __init__(self, in_features=30, encoder_h=128, output_h=128, use_input=False, use_ResNet=False, width=((3,3),(3,3),(3,3)), stride=((1,1),(1,1),(1,1)), dropout=0.0, targets=3, 
                 num_features_for_classification=int(15*64), multi_gpu=False, encoder_grad_frac=1.0):
        super().__init__(in_features, encoder_h)
        self.output_h = output_h
        if not isinstance(width, (list, tuple)):
            width = [width]
        if not isinstance(stride, (list, tuple)):
            stride = [stride]
        assert len(stride) == len(width)
        self.targets = targets
        self.num_features_for_classification = num_features_for_classification
        self.use_input = use_input
        self.use_ResNet = use_ResNet
        self.encoder = nn.Sequential()

        if self.use_input:
            self.encoder.add_module("Encoder_Input", nn.Sequential(
                nn.Conv1d(in_features, output_h, kernel_size=width[0][0], stride=width[0][0], padding=width[0][0] // 2),
                nn.Dropout(dropout),
                nn.Conv1d(output_h, output_h, kernel_size=(width[0][0]-1), stride=(width[0][0]-1), padding=(width[0][0]-1)// 2),
                nn.Dropout(dropout),
                nn.BatchNorm1d(output_h),
                nn.GELU(),
            ))

        if self.use_ResNet:
            for i, (_width, downsample) in enumerate(zip(width, stride)):
                self.encoder.add_module("Encoder_ResBlock{}".format(i), nn.Sequential(
                    ResBlockBN(output_h, output_h, res_width=_width, res_stride=downsample)
                ))
        
        self.encoder.add_module("Encoder_Output", nn.Sequential(
            nn.Conv1d(output_h, output_h, kernel_size=3, stride=3, padding=3 // 2),
            nn.Dropout(dropout),
            nn.Conv1d(output_h, int(output_h/2), kernel_size=2, stride=2, padding=2 // 2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(int(output_h/2)),
            nn.GELU(),
        ))
        self.encoder.add_module("Encoder_Cls", nn.Sequential(Flatten(), 
                                 nn.Linear(self.num_features_for_classification, self.targets))) 
    
    def forward(self, x):
        if self.use_input:
            x = self.encoder.Encoder_Input(x)
        if self.use_ResNet:
            x = self.encoder.Encoder_ResBlock0(x)
        
        features = self.encoder.Encoder_Output(x)
        logits = self.encoder.Encoder_Cls(features)

        return logits, features

class ConvEncoder_OutputClsHeavyFeaTL(_BENDREncoder):
    def __init__(self, in_features=30, encoder_h=128, output_h=128, use_input=False, use_ResNet=False, width=((3,3),(3,3),(3,3)), stride=((1,1),(1,1),(1,1)), dropout=0.0, targets=3, cls_heavy = 128,
                 num_features_for_classification=int(15*64), multi_gpu=False, encoder_grad_frac=1.0):
        super().__init__(in_features, encoder_h)
        self.output_h = output_h
        if not isinstance(width, (list, tuple)):
            width = [width]
        if not isinstance(stride, (list, tuple)):
            stride = [stride]
        assert len(stride) == len(width)
        self.targets = targets
        self.num_features_for_classification = num_features_for_classification
        self.use_input = use_input
        self.use_ResNet = use_ResNet
        self.cls_heavy = cls_heavy
        self.encoder = nn.Sequential()

        if self.use_input:
            self.encoder.add_module("Encoder_Input", nn.Sequential(
                nn.Conv1d(in_features, output_h, kernel_size=width[0][0], stride=width[0][0], padding=width[0][0] // 2),
                nn.Dropout(dropout),
                nn.Conv1d(output_h, output_h, kernel_size=(width[0][0]-1), stride=(width[0][0]-1), padding=(width[0][0]-1)// 2),
                nn.Dropout(dropout),
                nn.BatchNorm1d(output_h),
                nn.GELU(),
            ))

        if self.use_ResNet:
            for i, (_width, downsample) in enumerate(zip(width, stride)):
                self.encoder.add_module("Encoder_ResBlock{}".format(i), nn.Sequential(
                    ResBlockBN(output_h, output_h, res_width=_width, res_stride=downsample)
                ))
        
        self.encoder.add_module("Encoder_Output", nn.Sequential(
            nn.Conv1d(output_h, output_h, kernel_size=3, stride=3, padding=3 // 2),
            nn.Dropout(dropout),
            nn.Conv1d(output_h, int(output_h/2), kernel_size=2, stride=2, padding=2 // 2),
            nn.Dropout(dropout),
            nn.BatchNorm1d(int(output_h/2)),
            nn.GELU(),
        ))
        self.encoder.add_module("Encoder_Cls", nn.Sequential(Flatten(), 
                                 nn.Linear(self.num_features_for_classification, self.cls_heavy), 
                                 nn.Linear(self.cls_heavy, self.targets))) 
    
    def forward(self, x):
        if self.use_input:
            x = self.encoder.Encoder_Input(x)
        if self.use_ResNet:
            x = self.encoder.Encoder_ResBlock0(x)
        
        features = self.encoder.Encoder_Output(x)
        logits = self.encoder.Encoder_Cls(features)

        return logits, features


class ResEncoderfinetune(nn.Module):
    """
    The pretext task based on the designed ResEncoder
    """
    def __init__(self, encoder, encoder_output, output_h=128, targets=3, 
                 multi_gpu=False, encoder_grad_frac=1.0,):
        
        if multi_gpu:
            encoder = nn.DataParallel(encoder)
        if encoder_grad_frac < 1:
            encoder.register_backward_hook(lambda module, in_grad, out_grad:
                                           tuple(encoder_grad_frac * ig for ig in in_grad))
        super(ResEncoderfinetune, self).__init__()
        self.output_h = output_h
        self.targets = targets

        self.encoder = encoder
        self.encoder_output = encoder_output
        self.model = nn.Sequential(encoder, encoder_output)
            
    def forward(self, x):
        
        return self.model(x)
