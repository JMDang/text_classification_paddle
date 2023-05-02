#set -x
#在LD_LIBRARY_PATH中添加cuda库的路径
#export LD_LIBRARY_PATH=/home/work/cuda-9.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/work/cuda-10.2/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/work/cuda-10.2/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/work/cudnn/cudnn_v7.6/cuda/lib64:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/home/work/cuda-9.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH
#在LD_LIBRARY_PATH中添加cudnn库的路径
#export LD_LIBRARY_PATH=/home/work/cudnn/cudnn7.1.4/lib64:$LD_LIBRARY_PATH
#需要先下载NCCL，然后在LD_LIBRARY_PATH中添加NCCL库的路径
export LD_LIBRARY_PATH=/home/work/nccl_2.3.5/lib:$LD_LIBRARY_PATH
#如果FLAGS_sync_nccl_allreduce为1，则会在allreduce_op_handle中调用cudaStreamSynchronize（nccl_stream），这种模式在某些情况下可以获得更好的性能
export FLAGS_sync_nccl_allreduce=1
#是否是分布式训练，0标识是分布式，1标识是单机
export PADDLE_IS_LOCAL=1
#export PADDLE_USE_GPU=1
#表示分配的显存块占GPU总可用显存大小的比例，范围[0,1]
export FLAGS_fraction_of_gpu_memory_to_use=0.5
#选择要使用的GPU
export CUDA_VISIBLE_DEVICES=5
