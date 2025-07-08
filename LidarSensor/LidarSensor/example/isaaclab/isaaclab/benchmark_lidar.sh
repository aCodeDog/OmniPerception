#!/bin/bash

# Benchmark script to compare performance with and without LiDAR sensor

NUM_ENVS=1024
BENCHMARK_STEPS=1000

echo "============================================"
echo "LiDAR Sensor Performance Benchmark"
echo "============================================"
echo "Number of environments: $NUM_ENVS"
echo "Benchmark steps: $BENCHMARK_STEPS"
echo ""

echo "Running benchmark WITHOUT LiDAR sensor..."
echo "--------------------------------------------"
./isaaclab.sh -p scripts/examples/simple_lidar_integration.py \
    --num_envs $NUM_ENVS \
    --benchmark_steps $BENCHMARK_STEPS \
    --headless

echo ""
echo "Running benchmark WITH LiDAR sensor..."
echo "--------------------------------------------"
./isaaclab.sh -p scripts/examples/simple_lidar_integration.py \
    --num_envs $NUM_ENVS \
    --benchmark_steps $BENCHMARK_STEPS \
    --enable_lidar \
    --headless

echo ""
echo "Benchmark completed!"
