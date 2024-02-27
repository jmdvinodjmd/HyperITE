#!/bin/bash
#SBATCH --time=1
eval "$(conda shell.bash hook)"
conda activate base

#######################
# Experiments with 3 datasets
######################
dataset=IHDP
project=HyperITE-IHDP-Final
for seed in 1 2 3 4 5 6 7 8 9 10; do  # {11..1000}; do
    echo "runnning --- $seed"
    python run_learner.py --model SLearner --dataset $dataset --project $project --random-seed $seed
    python run_learner.py --model HyperSLearner --emb-dim1 1 --dataset $dataset --project $project --random-seed $seed --hn-drop-rate1 0.05
    
    python run_learner.py --model TLearner --dataset $dataset --project $project --random-seed $seed
    python run_learner.py --model HyperTLearner --dataset $dataset --project $project --hypernet1 all --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed

    python run_learner.py --model TARNet --dataset $dataset --project $project --random-seed $seed
    python run_learner.py --model HyperTARNet --dataset $dataset --project $project --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed

    python run_learner.py --model MitNet --dataset $dataset --project $project --random-seed $seed
    python run_learner.py --model HyperMitNet --dataset $dataset --project $project --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed

    python run_learner.py --model SNet --dataset $dataset --project $project --random-seed $seed 
    python run_learner.py --model HyperSNet --dataset $dataset --project $project --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed
    
    python run_learner.py --model DRLearner --dataset $dataset --project $project --random-seed $seed  --nfold 5
    python run_learner.py --model HyperDRLearnerPartial --dataset $dataset --project $project --hn-drop-rate1 0.05 --emb-dim1 16 --random-seed $seed --nfold 5
    
    python run_learner.py --model RALearner --dataset $dataset --project $project --random-seed $seed --nfold 5
    python run_learner.py --model HyperRALearner --dataset $dataset --project $project --emb-dim1 8 --random-seed $seed --hn-drop-rate1 0.05 --nfold 5
    
    python run_learner.py --model FlexTENet --dataset $dataset --project $project --random-seed $seed
done


dataset=twins
for n in 500 1000 2000 5000; do
    for p in 0.1; do      #0.1 0.25 0.5 0.75 0.9
        project='HyperITE-twins-Final-'$n'-'$p
        echo "runnning --- $project"
        for seed in 1 2 3 4 5 6 7 8 9 10; do  # 1 2 3 4 5 6 7 8 9 10
            echo "runnning --- $seed"
            python run_learner.py --model SLearner --dataset $dataset --n $n --p $p --project $project --random-seed $seed
            python run_learner.py --model HyperSLearner --emb-dim1 1 --dataset $dataset --n $n --p $p --project $project --random-seed $seed --hn-drop-rate1 0.05
            
            python run_learner.py --model TLearner --dataset $dataset --n $n --p $p --project $project --random-seed $seed
            python run_learner.py --model HyperTLearner --dataset $dataset --n $n --p $p --project $project --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed

            python run_learner.py --model TARNet --dataset $dataset --n $n --p $p --project $project --random-seed $seed
            python run_learner.py --model HyperTARNet --dataset $dataset --n $n --p $p --project $project --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed

            python run_learner.py --model MitNet --dataset $dataset --n $n --p $p --project $project --random-seed $seed
            # python run_learner.py --model HyperMitNet --dataset $dataset --n $n --p $p --project $project --emb-dim1 1 --hn-drop-rate1 0.05 --hypernet1 split --random-seed $seed # (600, alpha)
            python run_learner.py --model HyperMitNet --dataset $dataset --n $n --p $p --project $project --emb-dim1 8 --hn-drop-rate1 0.05 --hypernet1 split --random-seed $seed # 5000, (300, alpha)

            python run_learner.py --model SNet --dataset $dataset --n $n --p $p --project $project --random-seed $seed --exp-name corrected
            python run_learner.py --model HyperSNet --dataset $dataset --n $n --p $p --project $project --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed --exp-name corrected
            
            python run_learner.py --model DRLearner --dataset $dataset --n $n --p $p --project $project --random-seed $seed  --nfold 5 --exp-name 5fold
            python run_learner.py --model HyperDRLearnerPartial --dataset $dataset --n $n --p $p --project $project --hn-drop-rate1 0.05 --emb-dim1 8 --random-seed $seed --nfold 5
            
            python run_learner.py --model RALearner --dataset $dataset --n $n --p $p --project $project --random-seed $seed --nfold 5
            python run_learner.py --model HyperRALearner --dataset $dataset --n $n --p $p --project $project --emb-dim1 8 --random-seed $seed --hn-drop-rate1 0.05 --nfold 5 --exp-name corrected
            
            python run_learner.py --model FlexTENet --dataset $dataset --n $n --p $p --project $project --random-seed $seed
        done
    done
done


dataset=acic
for size in 500 1000 2000 4000; do

    project='HyperITE-ACIC-Final-'$size
    echo "runnning --- $project"

    for seed in 1 2 3 4 5 6 7 8 9 10; do

        echo "runnning --- $seed $dataset $size"

        python run_learner.py --model SLearner --dataset $dataset --data-size $size --project $project --random-seed $seed
        python run_learner.py --model HyperSLearner --emb-dim1 1 --hn-drop-rate1 0.05 --dataset $dataset --data-size $size --project $project --random-seed $seed

        python run_learner.py --model TLearner --dataset $dataset --data-size $size --project $project --random-seed $seed
        python run_learner.py --model HyperTLearner --dataset $dataset --data-size $size --project $project --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed

        python run_learner.py --model TARNet --dataset $dataset --data-size $size --project $project --random-seed $seed
        python run_learner.py --model HyperTARNet --dataset $dataset --data-size $size --project $project --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed 

        python run_learner.py --model MitNet --dataset $dataset --data-size $size --project $project --random-seed $seed
        python run_learner.py --model HyperMitNet --dataset $dataset --data-size $size --project $project --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed 

        python run_learner.py --model SNet --dataset $dataset --data-size $size --project $project --random-seed $seed
        python run_learner.py --model HyperSNet --dataset $dataset --data-size $size --project $project --emb-dim1 16 --hn-drop-rate1 0.05 --random-seed $seed
        
        python run_learner.py --model DRLearner --dataset $dataset --data-size $size --project $project --random-seed $seed --nfold 5
        python run_learner.py --model HyperDRLearnerPartial --dataset $dataset --data-size $size --project $project --hn-drop-rate1 0.05 --emb-dim1 32 --random-seed $seed --nfold 5
        
        python run_learner.py --model RALearner --dataset $dataset --data-size $size --project $project --random-seed $seed --nfold 5
        python run_learner.py --model HyperRALearner --dataset $dataset --data-size $size --project $project --emb-dim1 8 --random-seed $seed --hn-drop-rate1 0.05 --nfold 5
        
        python run_learner.py --model FlexTENet --dataset $dataset --data-size $size --project $project --random-seed $seed

    done
done


#######################
# Embedding effect
######################
dataset=IHDP
project=HyperITE-IHDP1024-extra
for seed in 1 2 3 4 5 6 7 8 9 10; do
    echo "runnning --- $seed"

    python run_learner.py --model HyperTLearner --dataset $dataset --project $project --hypernet1 split --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed --exp-name split8
    python run_learner.py --model HyperTLearner --dataset $dataset --project $project --hypernet1 split --emb-dim1 16 --hn-drop-rate1 0.05 --random-seed $seed --exp-name split16
    python run_learner.py --model HyperTLearner --dataset $dataset --project $project --hypernet1 split --emb-dim1 32 --hn-drop-rate1 0.05 --random-seed $seed --exp-name split32
    
    python run_learner.py --model HyperTARNet --dataset $dataset --project $project --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed --hypernet1 split --exp-name split8
    python run_learner.py --model HyperTARNet --dataset $dataset --project $project --emb-dim1 16 --hn-drop-rate1 0.05 --random-seed $seed --hypernet1 split --exp-name split16
    python run_learner.py --model HyperTARNet --dataset $dataset --project $project --emb-dim1 32 --hn-drop-rate1 0.05 --random-seed $seed --hypernet1 split --exp-name split32

    python run_learner.py --model HyperTARNet --dataset $dataset --project $project --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed --hypernet1 all --exp-name 8
    python run_learner.py --model HyperTARNet --dataset $dataset --project $project --emb-dim1 16 --hn-drop-rate1 0.05 --random-seed $seed --hypernet1 all --exp-name 16
    python run_learner.py --model HyperTARNet --dataset $dataset --project $project --emb-dim1 32 --hn-drop-rate1 0.05 --random-seed $seed --hypernet1 all --exp-name 32

    python run_learner.py --model HyperTLearner --dataset $dataset --project $project --hypernet1 all --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed --exp-name 8
    python run_learner.py --model HyperTLearner --dataset $dataset --project $project --hypernet1 all --emb-dim1 16 --hn-drop-rate1 0.05 --random-seed $seed --exp-name 16
    python run_learner.py --model HyperTLearner --dataset $dataset --project $project --hypernet1 all --emb-dim1 32 --hn-drop-rate1 0.05 --random-seed $seed --exp-name 32

done

dataset=acic
size=1000
project=HyperITE-acic1024-1K-extra
for seed in 1 2 3 4 5 6 7 8 9 10; do

    echo "runnning --- $seed $dataset"

    python run_learner.py --model HyperTLearner --dataset $dataset --data-size $size --project $project --hypernet1 split --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed --exp-name split8
    python run_learner.py --model HyperTLearner --dataset $dataset --data-size $size --project $project --hypernet1 split --emb-dim1 16 --hn-drop-rate1 0.05 --random-seed $seed --exp-name split16
    python run_learner.py --model HyperTLearner --dataset $dataset --data-size $size --project $project --hypernet1 split --emb-dim1 32 --hn-drop-rate1 0.05 --random-seed $seed --exp-name split32

    python run_learner.py --model HyperTARNet --dataset $dataset --data-size $size --project $project --hypernet1 split --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed --exp-name split8
    python run_learner.py --model HyperTARNet --dataset $dataset --data-size $size --project $project --hypernet1 split --emb-dim1 16 --hn-drop-rate1 0.05 --random-seed $seed --exp-name split16
    python run_learner.py --model HyperTARNet --dataset $dataset --data-size $size --project $project --hypernet1 split --emb-dim1 32 --hn-drop-rate1 0.05 --random-seed $seed --exp-name split32

    python run_learner.py --model HyperTLearner --dataset $dataset --data-size $size --project $project --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed --exp-name 8
    python run_learner.py --model HyperTLearner --dataset $dataset --data-size $size --project $project --emb-dim1 16 --hn-drop-rate1 0.05 --random-seed $seed --exp-name 16
    python run_learner.py --model HyperTLearner --dataset $dataset --data-size $size --project $project --emb-dim1 32 --hn-drop-rate1 0.05 --random-seed $seed --exp-name 32

    python run_learner.py --model HyperTARNet --dataset $dataset --data-size $size --project $project --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed --exp-name 8
    python run_learner.py --model HyperTARNet --dataset $dataset --data-size $size --project $project --emb-dim1 16 --hn-drop-rate1 0.05 --random-seed $seed --exp-name 16
    python run_learner.py --model HyperTARNet --dataset $dataset --data-size $size --project $project --emb-dim1 32 --hn-drop-rate1 0.05 --random-seed $seed --exp-name 32

done

dataset=twins
for n in 1000; do   #500 1000 2000 5000 None
    for p in 0.1; do      #0.1 0.25 0.5 0.75 0.9
        project='HyperITE-twins-1K-extra'
        echo "runnning --- $project"
        for seed in 1 2 3 4 5 6 7 8 9 10; do  # 1 2 3 4 5 6 7 8 9 10
            echo "runnning --- $seed"
        
            python run_learner.py --model HyperTLearner --dataset $dataset --n $n --p $p --project $project --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed --exp-name 8
            python run_learner.py --model HyperTLearner --dataset $dataset --n $n --p $p --project $project --emb-dim1 16 --hn-drop-rate1 0.05 --random-seed $seed --exp-name 16
            python run_learner.py --model HyperTLearner --dataset $dataset --n $n --p $p --project $project --emb-dim1 32 --hn-drop-rate1 0.05 --random-seed $seed --exp-name 32

            python run_learner.py --model HyperTARNet --dataset $dataset --n $n --p $p --project $project --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed --exp-name 8
            python run_learner.py --model HyperTARNet --dataset $dataset --n $n --p $p --project $project --emb-dim1 16 --hn-drop-rate1 0.05 --random-seed $seed --exp-name 16
            python run_learner.py --model HyperTARNet --dataset $dataset --n $n --p $p --project $project --emb-dim1 32 --hn-drop-rate1 0.05 --random-seed $seed --exp-name 32

            python run_learner.py --model HyperTLearner --dataset $dataset --n $n --p $p --project $project --hypernet1 split --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed --exp-name 8
            python run_learner.py --model HyperTLearner --dataset $dataset --n $n --p $p --project $project --hypernet1 split --emb-dim1 16 --hn-drop-rate1 0.05 --random-seed $seed --exp-name 16
            python run_learner.py --model HyperTLearner --dataset $dataset --n $n --p $p --project $project --hypernet1 split --emb-dim1 32 --hn-drop-rate1 0.05 --random-seed $seed --exp-name 32

            python run_learner.py --model HyperTARNet --dataset $dataset --n $n --p $p --project $project --hypernet1 split --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed --exp-name 8
            python run_learner.py --model HyperTARNet --dataset $dataset --n $n --p $p --project $project --hypernet1 split --emb-dim1 16 --hn-drop-rate1 0.05 --random-seed $seed --exp-name 16
            python run_learner.py --model HyperTARNet --dataset $dataset --n $n --p $p --project $project --hypernet1 split --emb-dim1 32 --hn-drop-rate1 0.05 --random-seed $seed --exp-name 32

        done
    done
done


########################
# Hypernetwork type effect
#######################
dataset=IHDP
project=HyperITE-IHDP1024-extra
for seed in 1 2 3 4 5 6 7 8 9 10; do
    echo "runnning --- $seed"

    python run_learner.py --model HyperTLearner --dataset $dataset --project $project --hypernet1 all --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed --exp-name all8
    python run_learner.py --model HyperTLearner --dataset $dataset --project $project --hypernet1 layerwise --emb-dim1 16 --hn-drop-rate1 0.05 --random-seed $seed --exp-name layerwise
    python run_learner.py --model HyperTLearner --dataset $dataset --project $project --hypernet1 chunking --emb-dim1 16 --hn-drop-rate1 0.05 --random-seed $seed --exp-name chunking
    
    python run_learner.py --model HyperTARNet --dataset $dataset --project $project --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed --hypernet1 all --exp-name all8
    python run_learner.py --model HyperTARNet --dataset $dataset --project $project --emb-dim1 16 --hn-drop-rate1 0.05 --random-seed $seed --hypernet1 layerwise --exp-name layerwise
    python run_learner.py --model HyperTARNet --dataset $dataset --project $project --emb-dim1 16 --hn-drop-rate1 0.05 --random-seed $seed --hypernet1 chunking --exp-name chunking

done

dataset=acic
size=1000
project=HyperITE-acic1024-1K-extra
for seed in 1 2 3 4 5 6 7 8 9 10; do

    echo "runnning --- $seed $dataset"

    python run_learner.py --model HyperTLearner --dataset $dataset --data-size $size --project $project --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed --exp-name 8
    python run_learner.py --model HyperTLearner --dataset $dataset --data-size $size --project $project --emb-dim1 16 --hn-drop-rate1 0.05 --random-seed $seed --hypernet1 layerwise --exp-name layerwise
    python run_learner.py --model HyperTLearner --dataset $dataset --data-size $size --project $project --emb-dim1 16 --hn-drop-rate1 0.05 --random-seed $seed --hypernet1 chunking --exp-name chunking

    python run_learner.py --model HyperTARNet --dataset $dataset --data-size $size --project $project --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed --exp-name 8
    python run_learner.py --model HyperTARNet --dataset $dataset --data-size $size --project $project --emb-dim1 16 --hn-drop-rate1 0.05 --random-seed $seed --hypernet1 layerwise --exp-name layerwise
    python run_learner.py --model HyperTARNet --dataset $dataset --data-size $size --project $project --emb-dim1 16 --hn-drop-rate1 0.05 --random-seed $seed --hypernet1 chunking --exp-name chunking

done

dataset=twins
for n in 1000; do   #500 1000 2000 5000 None
    for p in 0.1; do      #0.1 0.25 0.5 0.75 0.9
        project='HyperITE-twins-1K-extra'
        echo "runnning --- $project"
        for seed in 1 2 3 4 5 6 7 8 9 10; do
            echo "runnning --- $seed"
        
            python run_learner.py --model HyperTLearner --dataset $dataset --n $n --p $p --project $project --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed --exp-name 8
            python run_learner.py --model HyperTLearner --dataset $dataset --n $n --p $p --project $project --hypernet1 chunking --emb-dim1 16 --hn-drop-rate1 0.05 --random-seed $seed --exp-name chunking8
            python run_learner.py --model HyperTLearner --dataset $dataset --n $n --p $p --project $project --hypernet1 layerwise --emb-dim1 16 --hn-drop-rate1 0.05 --random-seed $seed --exp-name layerwise8

            python run_learner.py --model HyperTARNet --dataset $dataset --n $n --p $p --project $project --emb-dim1 8 --hn-drop-rate1 0.05 --random-seed $seed --exp-name 8
            python run_learner.py --model HyperTARNet --dataset $dataset --n $n --p $p --project $project --hypernet1 chunking --emb-dim1 16 --hn-drop-rate1 0.05 --random-seed $seed --exp-name chunking8
            python run_learner.py --model HyperTARNet --dataset $dataset --n $n --p $p --project $project --hypernet1 layerwise --emb-dim1 16 --hn-drop-rate1 0.05 --random-seed $seed --exp-name layerwise8

        done
    done
done

##################


###########
conda deactivate