#include <algorithm>
#include <vector>

#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

template <typename D>
__global__ void mean_var_reduce(int C, int HW, const D* data, D* x1, D* x2) {
  __shared__ D x1_temp[256];
  __shared__ D x2_temp[256];
  x1_temp[threadIdx.x] = 0;
  x2_temp[threadIdx.x] = 0;
  int warp_idx = threadIdx.x / 32;
  int c_idx = blockIdx.x * 8 + warp_idx;
  int x_loop = (HW + 31) / 32;
  int thread_offset = threadIdx.x % 32;
  for (int i = 0; i < x_loop; i++) {
    int idx = c_idx * HW + i * 32 + thread_offset;
    if (i * 32 + thread_offset < HW) {
      x1_temp[threadIdx.x] += data[idx];
      x2_temp[threadIdx.x] += data[idx] * data[idx];
    }
  }
  volatile D* vx1_temp = x1_temp;
  volatile D* vx2_temp = x2_temp;

  if (threadIdx.x % 32 < 16) {
    vx1_temp[threadIdx.x] += vx1_temp[threadIdx.x + 16];
    vx1_temp[threadIdx.x] += vx1_temp[threadIdx.x + 8];
    vx1_temp[threadIdx.x] += vx1_temp[threadIdx.x + 4];
    vx1_temp[threadIdx.x] += vx1_temp[threadIdx.x + 2];
    vx1_temp[threadIdx.x] += vx1_temp[threadIdx.x + 1];
    vx2_temp[threadIdx.x] += vx2_temp[threadIdx.x + 16];
    vx2_temp[threadIdx.x] += vx2_temp[threadIdx.x + 8];
    vx2_temp[threadIdx.x] += vx2_temp[threadIdx.x + 4];
    vx2_temp[threadIdx.x] += vx2_temp[threadIdx.x + 2];
    vx2_temp[threadIdx.x] += vx2_temp[threadIdx.x + 1];
  }

  if (threadIdx.x % 32 == 0) {
    if (c_idx < C) {
      x1[c_idx] += x1_temp[threadIdx.x];
      x2[c_idx] += x2_temp[threadIdx.x];
    }
  }
}

namespace caffe {
template <typename Dtype>
void BatchNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  if (skip_layer_) {
    return;
  }
  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count()/(channels_*bottom[0]->shape(0));

  if (use_global_stats_) {
    // use the stored mean/variance estimates.
    const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
      0 : 1 / this->blobs_[2]->cpu_data()[0];
    caffe_gpu_scale(variance_.count(), scale_factor,
        this->blobs_[0]->gpu_data(), mean_.mutable_gpu_data());
    caffe_gpu_scale(variance_.count(), scale_factor,
        this->blobs_[1]->gpu_data(), variance_.mutable_gpu_data());
  } else {
    Dtype* m = mean_.mutable_gpu_data();
    Dtype* v = variance_.mutable_gpu_data();
    for (int i = 0; i < num; i++) {
      mean_var_reduce<<<(channels_ + 7) / 8,256 >>> (channels_, spatial_dim, bottom_data + i * channels_ * spatial_dim, mean_.mutable_gpu_data(), variance_.mutable_gpu_data());
    }
    Dtype* host_m = mean_.mutable_cpu_data();
    Dtype* host_v = variance_.mutable_cpu_data();
    for (int i = 0; i < channels_; i++) {
      host_m[i] /= (num * spatial_dim);
      host_v[i] = host_v[i] / (num * spatial_dim) - host_m[i] * host_m[i];
    }
  }
  // subtract mean
  if (!skip_norm_) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
        batch_sum_multiplier_.gpu_data(), mean_.gpu_data(), 0.,
        num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
        spatial_dim, 1, -1, num_by_chans_.gpu_data(),
        spatial_sum_multiplier_.gpu_data(), 1., top_data);
  }

  if (!use_global_stats_) {
    // compute and save moving average
    this->blobs_[2]->mutable_cpu_data()[0] *= moving_average_fraction_;
    this->blobs_[2]->mutable_cpu_data()[0] += 1;
    caffe_gpu_axpby(mean_.count(), Dtype(1), mean_.gpu_data(),
        moving_average_fraction_, this->blobs_[0]->mutable_gpu_data());
    int m = bottom[0]->count()/channels_;
    Dtype bias_correction_factor = m > 1 ? Dtype(m)/(m-1) : 1;
    caffe_gpu_axpby(variance_.count(), bias_correction_factor,
        variance_.gpu_data(), moving_average_fraction_,
        this->blobs_[1]->mutable_gpu_data());
  }

  // normalize variance
  caffe_gpu_add_scalar(variance_.count(), eps_, variance_.mutable_gpu_data());
  caffe_gpu_sqrt(variance_.count(), variance_.gpu_data(),
      variance_.mutable_gpu_data());

  if (skip_norm_) {
    return;
  }
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), variance_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 0., temp_.mutable_gpu_data());
  caffe_gpu_div(temp_.count(), top_data, temp_.gpu_data(), top_data);
  // TODO(cdoersch): The caching is only needed because later in-place layers
  //                 might clobber the data.  Can we skip this if they won't?
  caffe_copy(x_norm_.count(), top_data,
      x_norm_.mutable_gpu_data());
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  Dtype* bottom_d = bottom[0]->mutable_gpu_diff();
  const Dtype* top_d = top[0]->gpu_diff();
  caffe_copy(bottom[0]->count(), top_d, bottom_d);

  if (skip_layer_ || skip_norm_) {
    return;
  }
  const Dtype* top_diff;
  if (bottom[0] != top[0]) {
    top_diff = top[0]->gpu_diff();
  } else {
    caffe_copy(x_norm_.count(), top[0]->gpu_diff(), x_norm_.mutable_gpu_diff());
    top_diff = x_norm_.gpu_diff();
  }
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  if (use_global_stats_) {
    caffe_gpu_div(temp_.count(), top_diff, temp_.gpu_data(), bottom_diff);
    return;
  }
  const Dtype* top_data = x_norm_.gpu_data();
  int num = bottom[0]->shape()[0];
  int spatial_dim = bottom[0]->count()/(channels_*bottom[0]->shape(0));
  // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
  //
  // dE(Y)/dX =
  //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
  //     ./ sqrt(var(X) + eps)
  //
  // where \cdot and ./ are hadamard product and elementwise division,
  // respectively, dE/dY is the top diff, and mean/var/sum are all computed
  // along all dimensions except the channels dimension.  In the above
  // equation, the operations allow for expansion (i.e. broadcast) along all
  // dimensions except the channels dimension where required.

  // sum(dE/dY \cdot Y)
  caffe_gpu_mul(temp_.count(), top_data, top_diff, bottom_diff);
  caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
      bottom_diff, spatial_sum_multiplier_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
      num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
      mean_.mutable_gpu_data());

  // reshape (broadcast) the above
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), mean_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 0., bottom_diff);

  // sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_mul(temp_.count(), top_data, bottom_diff, bottom_diff);

  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
      top_diff, spatial_sum_multiplier_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
      num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
      mean_.mutable_gpu_data());
  // reshape (broadcast) the above to make
  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), mean_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * channels_,
      spatial_dim, 1, 1., num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 1., bottom_diff);

  // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
  caffe_gpu_axpby(temp_.count(), Dtype(1), top_diff,
      Dtype(-1. / (num * spatial_dim)), bottom_diff);

  // note: temp_ still contains sqrt(var(X)+eps), computed during the forward
  // pass.
  caffe_gpu_div(temp_.count(), bottom_diff, temp_.gpu_data(), bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(BatchNormLayer);


}  // namespace caffe
