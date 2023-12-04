# On the Robustness of Normalizing Flows for Inverse Problems in Imaging

Official repo of On the Robustness of Normalizing Flows for Inverse Problems in Imaging (in ICCV 2023)

by [Seongmin Hong](https://smhongok.github.io/), [Inbum Park](https://inbumpark.github.io/), and [Se Young Chun](https://icl.snu.ac.kr/pi).

Links: [Project webpage](https://smhongok.github.io/robustness.html), [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Hong_On_the_Robustness_of_Normalizing_Flows_for_Inverse_Problems_in_ICCV_2023_paper.pdf), [Supp](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Hong_On_the_Robustness_ICCV_2023_supplemental.pdf), [arXiv](https://arxiv.org/abs/2212.04319)


<br><br><br>

## To run the code
### Environment
```.bash
python pip install -r requirements.txt
```
### RRDB pretrained weights
Please download the pre-trained weights for RRDB and put them in the 'pretrained_weights' directory.

[RRDB_DF2K_4X.pth](https://drive.google.com/file/d/1597Gblp_lsPEsyBuD8YrqelSMFbauObS/view?usp=sharing)\
[RRDB_DF2K_8X.pth](https://drive.google.com/file/d/1nyggiyxTZLAEOAjV6_x8UEmdoMdI5ulp/view?usp=sharing)

They are originally from [SRFlow](https://github.com/andreas128/SRFlow)

### Train and test

To run train.py and test.py, please refer [examples.sh](examples.sh).

### Results
<p align="center">
  <img src="figs/6.png">
</p>

For more results, please see the [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Hong_On_the_Robustness_of_Normalizing_Flows_for_Inverse_Problems_in_ICCV_2023_paper.pdf) or visit the [Project webpage](https://smhongok.github.io/robustness.html).


### Abstract
Conditional normalizing flows can generate diverse image samples for solving inverse problems. Most normalizing flows for inverse problems in imaging employ the conditional affine coupling layer that can generate diverse images quickly. However, unintended severe artifacts are occasionally observed in the output of them. In this work, we address this critical issue by investigating the origins of these artifacts and proposing the conditions to avoid them. First of all, we empirically and theoretically reveal that these problems are caused by "exploding inverse" in the conditional affine coupling layer for certain out-of-distribution (OOD) conditional inputs. Then, we further validated that the probability of causing erroneous artifacts in pixels is highly correlated with a Mahalanobis distance-based OOD score for inverse problems in imaging. Lastly, based on our investigations, we propose a remark to avoid exploding inverse and then based on it, we suggest a simple remedy that substitutes the affine coupling layers with the modified rational quadratic spline coupling layers in normalizing flows, to encourage the robustness of generated image samples. Our experimental results demonstrated that our suggested methods effectively suppressed critical artifacts occurring in normalizing flows for super-resolution space generation and low-light image enhancement. 



## References
This code is heavily based on [SRFlow](https://github.com/andreas128/SRFlow), [NCSR](https://github.com/younggeun-kim/NCSR), and [FS-NCSR](https://github.com/dsshim0125/FS-NCSR).


## BibTeX

If our code is helpful for your research, please consider citing
```
@InProceedings{Hong_2023_ICCV,
    author    = {Hong, Seongmin and Park, Inbum and Chun, Se Young},
    title     = {On the Robustness of Normalizing Flows for Inverse Problems in Imaging},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {10745-10755}
}
```
