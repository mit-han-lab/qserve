// Inspired by TRT-LLM.
// Modified by Haotian Tang and Shang Yang.
// @article{lin2024qserve,
//   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving},
//   author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
//   journal={arXiv preprint arXiv:2405.04532},
//   year={2024}
// }
#include <torch/extension.h>

__global__ void computePaddingOffsets(int *paddingOffsets, const int *seqOffsets, int maxSeqLength)
{
    // The index of the sequence in the batch.
    int batchIdx = blockIdx.x;

    // The beginning of the sequence.
    int seqBegin = seqOffsets[batchIdx];
    // The offset to the 1st element of the next sequence.
    int seqEnd = seqOffsets[batchIdx + 1];
    // The length of the sequence.
    int seqLength = seqEnd - seqBegin;

    // The number of padded tokens in the previous sequences.
    int paddingOffset = batchIdx * maxSeqLength - seqBegin;

    // Iterate over the tokens to update the number of padded elements.
    for (int tokenIdx = threadIdx.x; tokenIdx < seqLength; tokenIdx += blockDim.x)
    {
        paddingOffsets[seqBegin + tokenIdx] = paddingOffset;
    }
}

torch::Tensor compute_padding_offsets(torch::Tensor &cu_seqlens,
                                      int max_seqlen, int tot_num_tokens)
{
    int batch_size = cu_seqlens.size(0) - 1;
    auto options =
        torch::TensorOptions().dtype(torch::kInt32).device(cu_seqlens.device());
    at::Tensor padding_offsets =
        torch::empty({tot_num_tokens}, options);
    computePaddingOffsets<<<batch_size, 256>>>(
        padding_offsets.data_ptr<int>(), cu_seqlens.data_ptr<int>(),
        max_seqlen);
    return padding_offsets;
}
