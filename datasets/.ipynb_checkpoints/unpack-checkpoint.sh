cd ./UniversalFakeDetect
ls | xargs -I pa sh -c "tar -zxvf pa; rm pa"
cd ../
echo "UniversalFakeDetect разархивирован"
cd ./GANGen-Detection
ls | xargs -I pa sh -c "tar -zxvf pa; rm pa"
cd ../
echo "GANGen-Detection разархивирован"
cd ./DiffusionForensics
ls | xargs -I pa sh -c "tar -zxvf pa; rm pa"
cd ../
echo "DiffusionForensics разархивирован"
cd ./Diffusion1kStep
ls | xargs -I pa sh -c "tar -zxvf pa; rm pa"
cd ../
echo "Diffusion1kStep разархивирован"
cd ./ForenSynths_train_val
tar -zxvf CNN_synth_testset.zip -C ../ForenSynths
echo "ForenSynths разархивирован"