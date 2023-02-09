#!/bin/bash

INFO='default'
EPOCHS=50
LR=0.0001
TRAINWITHFOLDS=false # set true only to run 5-fold cross-validation to find optimal metaparameters 



if $TRAINWITHFOLDS
then
  for FOLD in {0..4}
     do
      echo $'\nFold: '$((FOLD+1)) 'of' 5
      python train.py --epochs=$EPOCHS --lr=$LR --info=$INFO --fold-bool --fold=$FOLD 
  done
  python utils/merge_folds_results.py --lr=$LR --info=$INFO 

else
  python train.py --epochs=$EPOCHS --lr=$LR --info=$INFO 
fi 