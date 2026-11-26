#!/bin/bash
MAX_JOBS=16
TOTAL_GPUS=8
MAX_RETRIES=1
ROOT=assets_sigkdd/baseline_final

get_gpu_allocation() {
    local job_number=$1
    # Calculate which GPU to allocate based on the job number
    local gpu_id=$((job_number % TOTAL_GPUS))
    echo $gpu_id+1
}

check_jobs() {
    while true; do
        jobs_count=$(jobs -p | wc -l)
        if [ "$jobs_count" -lt "$MAX_JOBS" ]; then
            break
        fi
        sleep 1
    done
}

job_number=0


run_with_retry() {
    local script=$1
    local gpu_allocation=$2
    local attempt=0
    while [ $attempt -le $MAX_RETRIES ]; do
        # Run the Python script
        CUDA_VISIBLE_DEVICES=-1 python $script
        status=$?
        if [ $status -eq 0 ]; then
            echo "Script $script succeeded."
            break
        else
            echo "Script $script failed on attempt $attempt. Retrying..."
            ((attempt++))
        fi
    done
    if [ $attempt -gt $MAX_RETRIES ]; then
        echo "Script $script failed after $MAX_RETRIES attempts."
    fi
}

job_number=0

DATA=ACIC
for seed in 1 2 3 4 5 6 7 8 9 10; do
    check_jobs
    gpu_allocation=$(get_gpu_allocation $job_number)
    ((job_number++))
    run_with_retry "src/estimator_site.py --seed $seed --data $DATA --root $ROOT" "$gpu_allocation" &
done

for seed in 1 2 3 4 5 6 7 8 9 10; do
    check_jobs
    gpu_allocation=$(get_gpu_allocation $job_number)
    ((job_number++))
    run_with_retry "src/estimator_descn.py --seed $seed --data $DATA --mu0hat_w 0.3 --mu1hat_w 0.8 --h0_w 0.1 --h1_w 0.1 --root $ROOT" "$gpu_allocation" &
done

for seed in 1 2 3 4 5 6 7 8 9 10; do
    check_jobs
    gpu_allocation=$(get_gpu_allocation $job_number)
    ((job_number++))
    run_with_retry "src/estimator_escfr.py --seed $seed --data $DATA --gamma 0.01 --root $ROOT" "$gpu_allocation" &
done

for seed in 1 2 3 4 5 6 7 8 9 10; do
    check_jobs
    gpu_allocation=$(get_gpu_allocation $job_number)
    ((job_number++))
    run_with_retry "src/estimator_metalearner.py --seed $seed --data $DATA --model slearner --root $ROOT" "$gpu_allocation" &
done

for seed in 1 2 3 4 5 6 7 8 9 10; do
    check_jobs
    gpu_allocation=$(get_gpu_allocation $job_number)
    ((job_number++))
    run_with_retry "src/estimator_metalearner.py --seed $seed --data $DATA --model tlearner --root $ROOT" "$gpu_allocation" &
done

for seed in 1 2 3 4 5 6 7 8 9 10; do
    check_jobs
    gpu_allocation=$(get_gpu_allocation $job_number)
    ((job_number++))
    run_with_retry "src/estimator_metalearner.py --seed $seed --data $DATA --model ylearner --root $ROOT" "$gpu_allocation" &
done

for seed in 1 2 3 4 5 6 7 8 9 10; do
    check_jobs
    gpu_allocation=$(get_gpu_allocation $job_number)
    ((job_number++))
    run_with_retry "src/estimator_cfr.py --seed $seed --data $DATA --model cfr-wass --root $ROOT" "$gpu_allocation" &
done

for seed in 1 2 3 4 5 6 7 8 9 10; do
    check_jobs
    gpu_allocation=$(get_gpu_allocation $job_number)
    ((job_number++))
    run_with_retry "src/estimator_cfr.py --seed $seed --data $DATA --model cfr-mmd --root $ROOT" "$gpu_allocation" &
done


for seed in 1 2 3 4 5 6 7 8 9 10; do
    check_jobs
    gpu_allocation=$(get_gpu_allocation $job_number)
    ((job_number++))
    run_with_retry "src/estimator_gromov.py --seed $seed --data $DATA --model fgwpca --root $ROOT" "$gpu_allocation" &
done

# python src/estimator_ml.py --data $DATA --root $ROOT


# ========================== IHDP ==========================

DATA=IHDP
for seed in 1 2 3 4 5 6 7 8 9 10; do
    check_jobs
    gpu_allocation=$(get_gpu_allocation $job_number)
    ((job_number++))
    run_with_retry "src/estimator_site.py --dim_pddm 16 --dim_c 16 --dim_s 16 --pddm_w 1 --mpdm_w 1 --seed $seed --data $DATA --root $ROOT" "$gpu_allocation" &
done

for seed in 1 2 3 4 5 6 7 8 9 10; do
    check_jobs
    gpu_allocation=$(get_gpu_allocation $job_number)
    ((job_number++))
    run_with_retry "src/estimator_descn.py --seed $seed --data $DATA --mu0hat_w 0.5 --mu1hat_w 0.5 --h0_w 4 --h1_w 4 --root $ROOT" "$gpu_allocation" &
done

for seed in 1 2 3 4 5 6 7 8 9 10; do
    check_jobs
    gpu_allocation=$(get_gpu_allocation $job_number)
    ((job_number++))
    run_with_retry "src/estimator_escfr.py --seed $seed --data $DATA --gamma 0.1 --root $ROOT" "$gpu_allocation" &
done

for seed in 1 2 3 4 5 6 7 8 9 10; do
    check_jobs
    gpu_allocation=$(get_gpu_allocation $job_number)
    ((job_number++))
    run_with_retry "src/estimator_metalearner.py --seed $seed --data $DATA --model slearner --root $ROOT" "$gpu_allocation" &
done

for seed in 1 2 3 4 5 6 7 8 9 10; do
    check_jobs
    gpu_allocation=$(get_gpu_allocation $job_number)
    ((job_number++))
    run_with_retry "src/estimator_metalearner.py --seed $seed --data $DATA --model tlearner --root $ROOT" "$gpu_allocation" &
done

for seed in 1 2 3 4 5 6 7 8 9 10; do
    check_jobs
    gpu_allocation=$(get_gpu_allocation $job_number)
    ((job_number++))
    run_with_retry "src/estimator_metalearner.py --seed $seed --data $DATA --model ylearner --root $ROOT" "$gpu_allocation" &
done

for seed in 1 2 3 4 5 6 7 8 9 10; do
    check_jobs
    gpu_allocation=$(get_gpu_allocation $job_number)
    ((job_number++))
    run_with_retry "src/estimator_cfr.py --seed $seed --data $DATA --model cfr-wass --root $ROOT" "$gpu_allocation" &
done

for seed in 1 2 3 4 5 6 7 8 9 10; do
    check_jobs
    gpu_allocation=$(get_gpu_allocation $job_number)
    ((job_number++))
    run_with_retry "src/estimator_cfr.py --seed $seed --data $DATA --model cfr-mmd --root $ROOT" "$gpu_allocation" &
done

for seed in 1 2 3 4 5 6 7 8 9 10; do
    check_jobs
    gpu_allocation=$(get_gpu_allocation $job_number)
    ((job_number++))
    run_with_retry "src/estimator_gromov.py --seed $seed --data $DATA --model fgwpca --dim_backbone 16,16 --dim_task 32,32 --fgw_alpha 0.2 --lambda 2 --ot_scale 1.0 --pca_ratio 0.5 --root $ROOT" "$gpu_allocation" &
done


python src/estimator_ml.py --data ACIC --root $ROOT
python src/estimator_ml.py --data IHDP --root $ROOT
wait
exec 6>&-
