# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains content licensed by https://github.com/chaiyujin/glow-pytorch/blob/master/LICENSE

import torch
from torch import nn as nn

import models.modules
import models.modules.Permutations
from models.modules import flow, thops, FlowAffineCouplingsAblation, FlowAffineReparamCouplingsAblation, FlowBsplineCouplingsAblation, FlowRQsplineCouplingsAblation, StdConditional
from utils.util import opt_get


def getConditional(rrdbResults, position):
    img_ft = rrdbResults if isinstance(rrdbResults, torch.Tensor) else rrdbResults[position]
    return img_ft


class FlowStep(nn.Module):
    FlowPermutation = {
        "reverse": lambda obj, z, logdet, rev: (obj.reverse(z, rev), logdet),
        "shuffle": lambda obj, z, logdet, rev: (obj.shuffle(z, rev), logdet),
        "invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "squeeze_invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "resqueeze_invconv_alternating_2_3": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "resqueeze_invconv_3": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1GridAlign": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1SubblocksShuf": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1GridAlignIndepBorder": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
        "InvertibleConv1x1GridAlignIndepBorder4": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
    }

    def __init__(self, in_channels, hidden_channels,
                 actnorm_scale=1.0, flow_permutation="invconv", flow_coupling="additive", stdCondition=True,
                 LU_decomposed=False, opt=None, image_injector=None, idx=None, acOpt=None, normOpt=None, in_shape=None,
                 position=None):
        # check configures
        assert flow_permutation in FlowStep.FlowPermutation, \
            "float_permutation should be in `{}`".format(
                FlowStep.FlowPermutation.keys())
        super().__init__()
        self.flow_permutation_train = opt_get(opt, ['network_G', 'flow', 'permutation_train'], True)
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        self.image_injector = image_injector

        self.norm_type = normOpt['type'] if normOpt else 'ActNorm2d'
        self.position = normOpt['position'] if normOpt else None

        self.in_shape = in_shape
        self.position = position
        self.acOpt = acOpt
        self.in_channels = in_channels
        self.std_mode = stdCondition

        # 1. actnorm
        self.actnorm = models.modules.FlowActNorms.ActNorm2d(in_channels, actnorm_scale)

        # 2. permute
        if flow_permutation == "invconv":
            if self.flow_permutation_train:
                self.invconv = models.modules.Permutations.InvertibleConv1x1(
                    in_channels, LU_decomposed=LU_decomposed)
            else:
                self.invconv = models.modules.Permutations.InvertibleConv1x1(
                    in_channels, LU_decomposed=LU_decomposed, _train=False
                )

        # 3. coupling
        if flow_coupling == "CondAffineSeparatedAndCond":
            self.affine = models.modules.FlowAffineCouplingsAblation.CondAffineSeparatedAndCond(in_channels=in_channels,
                                                                                                opt=opt, std_mode=self.std_mode)
        elif flow_coupling == "CondAffineReparamSeparatedAndCond":
            self.affinereparam = models.modules.FlowAffineReparamCouplingsAblation.CondAffineReparamSeparatedAndCond(in_channels=in_channels,
                                                                                                opt=opt, std_mode=self.std_mode)
        elif flow_coupling == "CondBsplineSeparatedAndCond":
            self.bspline = models.modules.FlowBsplineCouplingsAblation.CondBsplineSeparatedAndCond(in_channels=in_channels,
                                                                                                opt=opt, std_mode=self.std_mode)
        elif flow_coupling == "CondRQsplineSeparatedAndCond":
            self.rqspline = models.modules.FlowRQsplineCouplingsAblation.CondRQsplineSeparatedAndCond(in_channels=in_channels,
                                                                                                opt=opt, std_mode=self.std_mode)
        elif flow_coupling == "noCoupling":
            pass
        else:
            raise RuntimeError("coupling not Found:", flow_coupling)
            
        # 4. std Condition
        #if stdCondition is True:
        #    self.std = models.modules.StdConditional.StdConditionalLayer(dim_out=in_channels)
        #else:
        #    self.std = None
        

    def forward(self, input, logdet=None, reverse=False, rrdbResults=None, std=None):
        if not reverse:
            return self.normal_flow(input, logdet, rrdbResults, std)
        else:
            return self.reverse_flow(input, logdet, rrdbResults, std)

    def normal_flow(self, z, logdet, rrdbResults=None, std=None):
        if self.flow_coupling == "bentIdentityPreAct":
            z, logdet = self.bentIdentPar(z, logdet, reverse=False)      
            
        # 1. actnorm
        if self.norm_type == "ConditionalActNormImageInjector":
            img_ft = getConditional(rrdbResults, self.position)
            z, logdet = self.actnorm(z, img_ft=img_ft, logdet=logdet, reverse=False)
        elif self.norm_type == "noNorm":
            pass
        else:
            z, logdet = self.actnorm(z, logdet=logdet, reverse=False)
        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, False)

        need_features = self.affine_need_features()
        # 3. coupling
        if need_features or self.flow_coupling in ["condAffine", "condFtAffine", "condNormAffine"]:
            img_ft = getConditional(rrdbResults, self.position)
            z, logdet = self.affine(input=z, logdet=logdet, reverse=False, ft=img_ft, std=std)
        elif self.flow_coupling == "CondAffineReparamSeparatedAndCond":
            img_ft = getConditional(rrdbResults, self.position)
            z, logdet = self.affinereparam(input=z, logdet=logdet, reverse=False, ft=img_ft, std=std)
        elif self.flow_coupling == "CondBsplineSeparatedAndCond":
            img_ft = getConditional(rrdbResults, self.position)
            z, logdet = self.bspline(input=z, logdet=logdet, reverse=False, ft=img_ft, std=std)
            
        elif self.flow_coupling == "CondRQsplineSeparatedAndCond":
            img_ft = getConditional(rrdbResults, self.position)
            z, logdet = self.rqspline(input=z, logdet=logdet, reverse=False, ft=img_ft, std=std)
        # 4. std Condition
        #if std is not None and self.std is not None:
        #    z, logdet = self.std(input=z, logdet=logdet, reverse=False, std=std)

        return z, logdet

    def reverse_flow(self, z, logdet, rrdbResults=None, std=None):

        need_features = self.affine_need_features()   
        
        # 0. std Condition
        #if std is not None and self.std is not None:
        #    z, logdet = self.std(input=z, logdet=logdet, reverse=True, std=std)

        # 1.coupling
        if need_features or self.flow_coupling in ["condAffine", "condFtAffine", "condNormAffine"]:
            img_ft = getConditional(rrdbResults, self.position)
            z, logdet = self.affine(input=z, logdet=logdet, reverse=True, ft=img_ft, std=std)
        elif self.flow_coupling == "CondAffineReparamSeparatedAndCond":
            img_ft = getConditional(rrdbResults, self.position)
            z, logdet = self.affinereparam(input=z, logdet=logdet, reverse=True, ft=img_ft, std=std)
        elif self.flow_coupling == "CondBsplineSeparatedAndCond":
            img_ft = getConditional(rrdbResults, self.position)
            z, logdet = self.bspline(input=z, logdet=logdet, reverse=True, ft=img_ft, std=std)
        elif self.flow_coupling == "CondRQsplineSeparatedAndCond":
            img_ft = getConditional(rrdbResults, self.position)
            z, logdet = self.rqspline(input=z, logdet=logdet, reverse=True, ft=img_ft, std=std)
        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, True)

        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet

    def affine_need_features(self):
        need_features = False
        try:
            need_features = self.affine.need_features
        except:
            pass
        return need_features
