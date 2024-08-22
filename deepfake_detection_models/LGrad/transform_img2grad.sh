#!/bin/bash

# sh ./transform_img2grad.sh 0 ../../datasets/ ./grads    
# nohup sh ./transform_img2grad.sh 0 ../../datasets/ ./grads > transform_img2grad.log 2>&1 &
# ps -ef | grep transform_img2grad.sh # (kill pid from: jovyan    <pid>  220314  0 23:34 pts/7    00:00:00 sh ./transform_img2grad.sh 0 ../../datasets/ ./grads)


Classes='0_real 1_fake'
GANmodelpath=$(cd $(dirname $0); pwd)/img2gad/stylegan/
Model=karras2019stylegan-bedrooms-256x256.pkl
Imgrootdir=$2
Saverootdir=$3


# # Valdatas='airplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor'
# Valdatas='horse car cat chair'
# Valrootdir=${Imgrootdir}/val/
# Savedir=$Saverootdir/val/

# for Valdata in $Valdatas
# do
#     for Class in $Classes
#     do
#         Imgdir=${Valdata}/${Class}
#         CUDA_VISIBLE_DEVICES=$1 /usr/bin/python $GANmodelpath/img2grad.py 1\
#             ${Valrootdir}${Imgdir} \
#             ${Savedir}${Imgdir}_grad \
#             ${GANmodelpath}networks/${Model} \
#             1
#     done
# done


# Traindatas='horse car cat chair'
# Trainrootdir=${Imgrootdir}/train/
# Savedir=$Saverootdir/train/
# for Traindata in $Traindatas
# do
#     for Class in $Classes
#     do
#         Imgdir=${Traindata}/${Class}
#         CUDA_VISIBLE_DEVICES=$1 /usr/bin/python $GANmodelpath/img2grad.py 1\
#             ${Trainrootdir}${Imgdir} \
#             ${Savedir}${Imgdir}_grad \
#             ${GANmodelpath}networks/${Model} \
#             1
#     done
# done


# Testdatas='DiffusionForensics/adm/imagenet UniversalFakeDetect/ldm_200_cfg UniversalFakeDetect/ldm_200 ForenSynths/progan/sofa DiffusionForensics/midjourney/bedroom DiffusionForensics/sdv2/bedroom DiffusionForensics/iddpm/bedroom ForenSynths/stylegan/cat Diffusion1kStep/ddpm/google-ddpm-bedroom-256 ForenSynths/cyclegan/summer UniversalFakeDetect/glide_50_27 GANGen-Detection/AttGAN Diffusion1kStep/ddpm/google-ddpm-church-256 ForenSynths/seeingdark ForenSynths/stargan DiffusionForensics/if/bedroom ForenSynths/imle ForenSynths/stylegan/bedroom ForenSynths/progan/bus UniversalFakeDetect/glide_100_10 ForenSynths/progan/cow ForenSynths/stylegan/car ForenSynths/progan/bird ForenSynths/crn ForenSynths/whichfaceisreal ForenSynths/cyclegan/apple ForenSynths/cyclegan/zebra DiffusionForensics/diff-stylegan/bedroom GANGen-Detection/BEGAN DiffusionForensics/pndm/bedroom UniversalFakeDetect/dalle UniversalFakeDetect/guided ForenSynths/san GANGen-Detection/MMDGAN ForenSynths/progan/person ForenSynths/cyclegan/winter Diffusion1kStep/ddpm/google-ddpm-celebahq-256 ForenSynths/progan/motorbike GANGen-Detection/STGAN ForenSynths/stylegan2/horse Diffusion1kStep/ddpm/google-ddpm-cifar10-32 ForenSynths/progan/dog ForenSynths/progan/boat Diffusion1kStep/DALLE DiffusionForensics/ddpm/bedroom Diffusion1kStep/guided-diffusion ForenSynths/progan/chair GANGen-Detection/SNGAN ForenSynths/deepfake DiffusionForensics/if/celebahq DiffusionForensics/stylegan_official/bedroom ForenSynths/gaugan GANGen-Detection/InfoMaxGAN ForenSynths/progan/car DiffusionForensics/sdv1/imagenet Diffusion1kStep/improved-diffusion DiffusionForensics/midjourney/celebahq ForenSynths/progan/airplane GANGen-Detection/CramerGAN Diffusion1kStep/midjourney UniversalFakeDetect/glide_100_27 DiffusionForensics/projectedgan/bedroom Diffusion1kStep/ddpm/google-ddpm-cat-256 DiffusionForensics/diff-projectedgan/bedroom ForenSynths/progan/cat DiffusionForensics/vqdiffusion/bedroom DiffusionForensics/ldm/bedroom ForenSynths/progan/bicycle ForenSynths/stylegan2/church ForenSynths/progan/horse ForenSynths/cyclegan/horse UniversalFakeDetect/ldm_100 DiffusionForensics/dalle2/celebahq ForenSynths/progan/sheep ForenSynths/progan/bottle ForenSynths/stylegan2/car ForenSynths/biggan ForenSynths/progan/train DiffusionForensics/adm/bedroom DiffusionForensics/dalle2/bedroom DiffusionForensics/sdv2/celebahq GANGen-Detection/RelGAN ForenSynths/stylegan2/cat ForenSynths/cyclegan/orange ForenSynths/progan/diningtable ForenSynths/progan/tvmonitor ForenSynths/progan/pottedplant GANGen-Detection/S3GAN'
Testdatas='UniversalFakeDetect/guided DiffusionForensics/sdv1/imagenet'
Testrootdir=${Imgrootdir}
Savedir=$Saverootdir/

for Testdata in $Testdatas
do
    for Class in $Classes
    do
        Imgdir=${Testdata}/${Class}
        CUDA_VISIBLE_DEVICES=$1 /home/jovyan/shares/SR006.nfs2/almas_deepfake_detection/env/almas-env/bin/python $GANmodelpath/img2grad.py 1\
            ${Testrootdir}${Imgdir} \
            ${Savedir}${Imgdir}_grad \
            ${GANmodelpath}networks/${Model} \
            64
    done
done

