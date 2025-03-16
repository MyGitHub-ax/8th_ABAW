export PYTHONPATH=$PYTHONPATH:/data/yanglongjiang/project/ABAW

#python3 visual/extract_vision_huggingfacev2.py --dataset=AVEC2013 --model_name=clip-vit-large-patch14 --feature_level=FRAME
#python3 visual/extract_vision_huggingfacev2.py --dataset=ABAW --model_name=clip-vit-large-patch14 --feature_level=FRAME

python3 audio/extract_audio_huggingface.py --dataset=ABAW --model_name=hubert-base-ls960 --feature_level=FRAME