#include <vector>
#include <math.h>
#include "caffe/layers/conv_layer.hpp"
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "cuda_profiler_api.h"

namespace caffe {
void* passed_gamma_ptr;
void* passed_beta_ptr;

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // simplify the normalize with using only gamma and beta.
  vector<float> gamma_save(this->conv_in_channels_, 0);
  vector<float> beta_save(this->conv_in_channels_, 0);
  if (this->norm_fusion_) {
    const Dtype* mean_cpu = this->mean_.cpu_data();
    const Dtype* var_cpu = this->var_.cpu_data();
    Dtype* gamma_cpu = this->gamma_.mutable_cpu_data();
    Dtype* beta_cpu = this->beta_.mutable_cpu_data();
    for (int i = 0; i < this->conv_in_channels_; i++) {
      beta_save[i] = beta_cpu[i];
      gamma_save[i] = gamma_cpu[i];
      beta_cpu[i] += -gamma_cpu[i] * mean_cpu[i] / var_cpu[i];
      gamma_cpu[i] *= 1 / var_cpu[i];
    }
  }
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      if (this->mean_var_fusion_ && !this->norm_fusion_) {
        LOG(FATAL) << "Cant reach here";
      }
      // x1 conv (filter size 1x1 conv) case
      else if (this->mean_var_fusion_ && this->norm_fusion_) {
        this->forward_gpu_gemm_mean_var_norm_fusion(bottom_data + n * this->bottom_dim_, weight,
            top_data + n * this->top_dim_, false, n);
      }
      // x2/blk conv case
      else if (!this->mean_var_fusion_ && this->norm_fusion_) {
        this->forward_gpu_gemm_norm_fusion(bottom_data + n * this->bottom_dim_, weight,
            top_data + n * this->top_dim_, false, n);
      }
      // no fusion
      else {
        this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
            top_data + n * this->top_dim_);
      }
      // no fusion support for bias yet
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
    // sum up X and X^2 to compute mean and variance to use in latter layers.
    // x_temp, x2_temp : each for E(x), E(x^2)
    if (this->mean_var_fusion_) {
      const Dtype* h_x_temp = this->x_accum_temp_.cpu_data();
      const Dtype* h_x2_temp = this->x2_accum_temp_.cpu_data();
      Dtype* mean_to_pass = this->mean_to_pass_.mutable_cpu_data();
      Dtype* var_to_pass = this->var_to_pass_.mutable_cpu_data();
      // gathers up the x_temp and x2_temp along bx (gpu) dimension.
      // currently we use hard-coded thread block size (channel-first dim)
      for (int ch = 0; ch < this->conv_out_channels_; ch++) {
        for (int bi = 0; bi < (this->conv_out_spatial_dim_ + 127) / 128; bi++) {
          mean_to_pass[ch] += h_x_temp[ch + bi * this->conv_out_channels_];
          var_to_pass[ch] += h_x2_temp[ch + bi * this->conv_out_channels_];
        }
      }
      // make mean and var (actually, 1/ sqrt(V(x)^2 + eps) from x_temp and x2_temp
      for (int ch = 0; ch < this->conv_out_channels_; ch++) {
        mean_to_pass[ch] /= (this->num_ * this->conv_out_spatial_dim_);
        var_to_pass[ch] = std::sqrt(var_to_pass[ch] / (this->num_ * this->conv_out_spatial_dim_) - mean_to_pass[ch] * mean_to_pass[ch] + 1e-5f);
      }
    }
    // rollback gamma beta (see line 16-17)
    if (this->norm_fusion_) {
      Dtype* gamma_cpu = this->gamma_.mutable_cpu_data();
      Dtype* beta_cpu = this->beta_.mutable_cpu_data();
      for (int i = 0; i < this->conv_in_channels_; i++) {
        beta_cpu[i] = beta_save[i];
        gamma_cpu[i] = gamma_save[i];
      }
    }
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    // no fusion case
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      if (!this->norm_fusion_ && !this->mean_var_fusion_) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        for (int n = 0; n < this->num_; ++n) {
          // gradient w.r.t. weight. Note that we will accumulate diffs.
          if (this->param_propagate_down_[0]) {
            this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
                top_diff + n * this->top_dim_, weight_diff);
          }
          // gradient w.r.t. bottom data, if necessary.
          if (propagate_down[i]) {
            this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
                bottom_diff + n * this->bottom_dim_);
          }
        }
      }
      // x2 and blk case
      else if (this->norm_fusion_ && !this->mean_var_fusion_) {
        // save the input of the relu layer
        const Dtype* relu_data = this->relu_inp_.gpu_data();
        const Dtype* bottom_data = bottom[i]->gpu_data();
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        int i_c = this->conv_in_channels_;
        int i_hw = this->conv_input_shape_.cpu_data()[1] * this->conv_input_shape_.cpu_data()[2];
        int i_chw = i_c * i_hw;

        cudaMemset(this->d_gamma_chw, 0, i_chw * sizeof(Dtype));
        cudaMemset(this->d_beta_chw, 0, i_chw * sizeof(Dtype));
        for (int n = 0; n < this->num_; ++n) {
          // gradient w.r.t. bottom data, if necessary.
          if (propagate_down[i]) {
            // x_relu, x_bn added
            this->backward_gpu_gemm_gather(top_diff + n * this->top_dim_, weight, relu_data + n * this->bottom_dim_, bottom_data + n * this->bottom_dim_,
                bottom_diff + n * this->bottom_dim_);
          }
          // gradient w.r.t. weight. Note that we will accumulate diffs.
          if (this->param_propagate_down_[0]) {
            this->weight_gpu_gemm_relu(relu_data + n * this->bottom_dim_,
                top_diff + n * this->top_dim_, weight_diff);
          }
        }
        this->backward_gpu_reduction(this->gamma_.mutable_gpu_diff(), const_cast<const Dtype*> (this->d_gamma_chw));
        this->backward_gpu_reduction(this->beta_.mutable_gpu_diff(),  const_cast<const Dtype*> (this->d_beta_chw));

        passed_gamma_ptr = static_cast<void *> (&this->gamma_);
        passed_beta_ptr = static_cast<void *> (&this->beta_);

        if (this->is_blk_) {
          // reduction C, H, W to C
          this->backward_gpu_reduction(this->gamma_.mutable_gpu_diff(), const_cast<const Dtype*> (this->d_gamma_chw));
          this->backward_gpu_reduction(this->beta_.mutable_gpu_diff(),  const_cast<const Dtype*> (this->d_beta_chw));
          // Ady + Bx + C = dx optimize
          caffe_gpu_compute_ABC(this->conv_in_channels_, this->num_ * this->conv_input_shape_.cpu_data()[1] * this->conv_input_shape_.cpu_data()[2], this->gamma_.gpu_data(), this->var_.gpu_data(), this->gamma_.gpu_diff(), this->beta_.gpu_diff(), this->mean_.gpu_data(), this->A, this->B, this->C);
          Dtype* bottom_diff_mutable = bottom[i]->mutable_gpu_diff();
          for (int n = 0; n < this->num_; ++n) {
            caffe_gpu_top_diff_transform(this->conv_in_channels_, this->conv_input_shape_.cpu_data()[1] * this->conv_input_shape_.cpu_data()[2], this->A, this->B, this->C, bottom_diff_mutable + n * this->bottom_dim_, bottom_data + n * this->bottom_dim_);
          }
        }
      }
      else if (this->norm_fusion_ && this->mean_var_fusion_) {
        int i_c = this->conv_in_channels_;
        int i_hw = this->conv_input_shape_.cpu_data()[1] * this->conv_input_shape_.cpu_data()[2];
        int i_chw = i_c * i_hw;
        const Dtype* bottom_data = bottom[i]->gpu_data();
        cudaMemset(this->d_gamma_chw, 0, i_chw * sizeof(Dtype));
        cudaMemset(this->d_beta_chw, 0, i_chw * sizeof(Dtype));

        const Dtype* relu_data = this->relu_inp_.gpu_data();
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        const Dtype* top_data = top[i]->gpu_data();
        // for 1x1 norm, we do first compute A, B, C per output channel where
        // var is sqrt(...)
        // A : r/var
        // B : - dr/(NHW * var)
        // C : - ( r * db + u * dr) / NHW * var
        // in order to do Ady + Bx_top + C in weight_gpu_gemm relu.
        Dtype* top_diff_mutable = top[i]->mutable_gpu_diff();
        Blob<Dtype> *passed_gamma = static_cast<Blob<Dtype>*> (passed_gamma_ptr);
        Blob<Dtype> *passed_beta = static_cast<Blob<Dtype>*> (passed_beta_ptr);

        caffe_gpu_compute_ABC(this->conv_out_channels_, this->num_ * this->conv_out_spatial_dim_, passed_gamma->gpu_data(), this->var_to_pass_.gpu_data(), passed_gamma->gpu_diff(), passed_beta->gpu_diff(), this->mean_to_pass_.gpu_data(), this->A, this->B, this->C);
        for (int n = 0; n < this->num_; ++n) {
          // do norm for top_diff, compute dx, save the normed top_diff to temp, gather d_beta, d_gamma
          // transform top_diff for every batch; maximize L2 cache effect
          caffe_gpu_top_diff_transform(this->conv_out_channels_, this->conv_out_spatial_dim_, this->A, this->B, this->C, top_diff_mutable + n * this->top_dim_, top_data + n * this->top_dim_);
          if (propagate_down[i]) {
            this->backward_gpu_gemm_gather(top_diff + n * this->top_dim_, weight, relu_data + n * this->bottom_dim_, bottom_data + n * this->bottom_dim_,
                bottom_diff + n * this->bottom_dim_, n);
          }
          if (this->param_propagate_down_[0]) {
            this->weight_gpu_gemm_relu(relu_data + n * this->bottom_dim_,
                top_diff + n * this->top_dim_, weight_diff);
          }
        }
        this->backward_gpu_reduction(this->gamma_.mutable_gpu_diff(), const_cast<const Dtype*> (this->d_gamma_chw));
        this->backward_gpu_reduction(this->beta_.mutable_gpu_diff(),  const_cast<const Dtype*> (this->d_beta_chw));
        caffe_gpu_compute_ABC(this->conv_in_channels_, this->num_ * this->conv_input_shape_.cpu_data()[1] * this->conv_input_shape_.cpu_data()[2], this->gamma_.gpu_data(), this->var_.gpu_data(), this->gamma_.gpu_diff(), this->beta_.gpu_diff(), this->mean_.gpu_data(), this->A, this->B, this->C);
        Dtype* bottom_diff_mutable = bottom[i]->mutable_gpu_diff();
        for (int n = 0; n < this->num_; ++n) {
          caffe_gpu_top_diff_transform(this->conv_in_channels_, this->conv_input_shape_.cpu_data()[1] * this->conv_input_shape_.cpu_data()[2], this->A, this->B, this->C, bottom_diff_mutable + n * this->bottom_dim_, bottom_data + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
